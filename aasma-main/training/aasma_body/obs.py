from typing import Any

import numpy as np
from rlgym.utils.common_values import BOOST_LOCATIONS
from rlgym.utils.gamestates import GameState
from rocket_learn.utils.batched_obs_builder import BatchedObsBuilder

from ._constants import POS, FW, ANG_VEL, LIN_VEL, IS_BOOST, DEMO, BOOST, ON_GROUND, HAS_FLIP, \
    UP, IS_MATE, IS_OPP, ACTIONS, IS_SELF


class AasmaObsBuilder(BatchedObsBuilder):
    def __init__(self, n_players=None, tick_skip=8):
        super().__init__()
        self.n_players = n_players
        self.demo_timers = None
        self.boost_timers = None
        self.current_state = None
        self.current_qkv = None
        self.current_mask = None
        self.tick_skip = tick_skip
        self._boost_locations = np.array(BOOST_LOCATIONS)
        self._invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
        self._norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)

    def _reset(self, initial_state: GameState) -> None:
        """
        Reset Demolition timers and Boost timers of the observation builder

        Args:
            initial_state (GameState)
        """
        self.demo_timers = np.zeros(len(initial_state.players))
        self.boost_timers = np.zeros(len(initial_state.boost_pads))

    @staticmethod
    def _quats_to_rot_mtx(quats: np.ndarray) -> np.ndarray:
        """
        [Adapted from rlgym.utils.math.quat_to_rot_mtx]

        Convert quaternion to rotation matrix

        Args:
            quats (np.ndarray): Quaternion array

        Returns:
            Numpy ndarray with rotation matrix
        """
        w = -quats[:, 0]
        x = -quats[:, 1]
        y = -quats[:, 2]
        z = -quats[:, 3]

        theta = np.zeros((quats.shape[0], 3, 3))

        norm = np.einsum("fq,fq->f", quats, quats)

        sel = norm != 0

        w = w[sel]
        x = x[sel]
        y = y[sel]
        z = z[sel]

        s = 1.0 / norm[sel]

        # front direction
        theta[sel, 0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[sel, 1, 0] = 2.0 * s * (x * y + z * w)
        theta[sel, 2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[sel, 0, 1] = 2.0 * s * (x * y - z * w)
        theta[sel, 1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[sel, 2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[sel, 0, 2] = 2.0 * s * (x * z + y * w)
        theta[sel, 1, 2] = 2.0 * s * (y * z - x * w)
        theta[sel, 2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

        return theta

    @staticmethod
    def convert_to_relative(q, kv):
        """
        Convert forward direction in keys_and_values array to relative direction according to x and y

        Args:
            q (np.ndarray): Queries
            kv (np.ndarray): Keys and Values
        """
        # kv[..., POS.start:LIN_VEL.stop] -= q[..., POS.start:LIN_VEL.stop]
        kv[..., POS] -= q[..., POS]
        forward = q[..., FW]
        theta = np.arctan2(forward[..., 0], forward[..., 1])
        theta = np.expand_dims(theta, axis=-1)
        ct = np.cos(theta)
        st = np.sin(theta)
        xs = kv[..., POS.start:ANG_VEL.stop:3]
        ys = kv[..., POS.start + 1:ANG_VEL.stop:3]
        # Use temp variables to prevent modifying original array
        nx = ct * xs - st * ys
        ny = st * xs + ct * ys
        kv[..., POS.start:ANG_VEL.stop:3] = nx  # x-components
        kv[..., POS.start + 1:ANG_VEL.stop:3] = ny  # y-components

    def batched_build_obs(self, encoded_states: np.ndarray):
        """
        Build observation

        - Observation array composition:
            [0,3]
            [3, 34] -> Boosts Info
            [34, 52] -> Ball Info
            [52, ...] -> Players Info

        - Selectors
            sel_players = [0 -> number_of_players]
            sel_ball = [number_of_players] (sel_players.stop)
            sel_boosts = [sel_ball + 1 -> end_of_array]

        - Main arrays
            queries -> [number_of_players, batch_size, 1, 32]
                32 is the game state info with the actions (see _constants.py for more info)

            keys_and_values -> [number_of_players, batch_size, number_of_entities, 24]
                24 is the game state info without the actions (see _constants.py for more info)

            mask -> [number_of_players, batch_size, number_of_entities]

            NOTE: number_of_entities = number of players + 1 ball + 34 boosts

        - Ball
            keys_and_values[:, :, sel_ball, 3] = 1
                3 represents the IS_BALL constant in _constants.py which we set to true

            ball_info = np.r_[POS, LIN_VEL, ANG_VEL] = [5 -> 8, 8 -> 11, 17 -> 20]
                nine positions representing the ball values

            keys_and_values[:, :, sel_ball, ball_info] = encoded_states[:, ball_start_index: ball_start_index + 9]
                9 positions representing the ball values

        - Boosts
            keys_and_values[:, :, sel_boosts, IS_BOOST] = 1

            keys_and_values[:, :, sel_boosts, POS] = self._boost_locations

            keys_and_values[:, :, sel_boosts, BOOST] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)
                Small boost pads give 12 (0.12) boost, while Big boost pads give 100 (1)
                (self._boost_locations[:, 2] > 72) -> Big boost pads have higher Z axis

            keys_and_values[:, :, sel_boosts, DEMO] = encoded_states[:, 3:3 + 34]
                [3, 34] -> Boosts Info
                DEMO here is like the cars (the boost is on cooldown or not)

        - Teams
            teams = encoded_states[0, players_start_index + 1::player_length]
                players_start_index = 52
                teams equals to the blue cars

            keys_and_values[:, :, :number_of_players, IS_MATE] = 1 - teams
                Sets Orange team players bool to true

            keys_and_values[:, :, :number_of_players, IS_OPP] = teams
                Sets Blue team players bool to true

        - Player (player ID identified by i)
                encoded_player = encoded_states[:, players_start_index + i * player_length:
                                                players_start_index + (i + 1) * player_length]
                    Get info of respective player
                    Then use it to set the reamining of the variables

        Args:
            encoded_states (np.ndarray)

        Returns:

        """
        ball_start_index = 3 + GameState.BOOST_PADS_LENGTH
        players_start_index = ball_start_index + GameState.BALL_STATE_LENGTH
        player_length = GameState.PLAYER_INFO_LENGTH

        n_players = (encoded_states.shape[1] - players_start_index) // player_length
        lim_players = n_players if self.n_players is None else self.n_players
        n_entities = lim_players + 1 + 34

        # SELECTORS
        sel_players = slice(0, lim_players)
        sel_ball = sel_players.stop
        sel_boosts = slice(sel_ball + 1, None)

        # MAIN ARRAYS
        q = np.zeros((n_players, encoded_states.shape[0], 1, 32))
        kv = np.zeros((n_players, encoded_states.shape[0], n_entities, 24))  # Keys and values are (mostly) shared
        m = np.zeros((n_players, encoded_states.shape[0], n_entities))  # Mask is shared

        # BALL
        kv[:, :, sel_ball, 3] = 1
        kv[:, :, sel_ball, np.r_[POS, LIN_VEL, ANG_VEL]] = encoded_states[:, ball_start_index: ball_start_index + 9]

        # BOOSTS
        kv[:, :, sel_boosts, IS_BOOST] = 1
        kv[:, :, sel_boosts, POS] = self._boost_locations
        kv[:, :, sel_boosts, BOOST] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)
        kv[:, :, sel_boosts, DEMO] = encoded_states[:, 3:3 + 34]

        # PLAYERS
        teams = encoded_states[0, players_start_index + 1::player_length]
        kv[:, :, :n_players, IS_MATE] = 1 - teams  # Default team is blue
        kv[:, :, :n_players, IS_OPP] = teams
        for i in range(n_players):
            encoded_player = encoded_states[:,
                             players_start_index + i * player_length: players_start_index + (i + 1) * player_length]

            kv[i, :, i, IS_SELF] = 1
            kv[:, :, i, POS] = encoded_player[:, 2: 5]
            kv[:, :, i, LIN_VEL] = encoded_player[:, 9: 12]
            quats = encoded_player[:, 5: 9]
            rot_mtx = self._quats_to_rot_mtx(quats)
            kv[:, :, i, FW] = rot_mtx[:, :, 0]
            kv[:, :, i, UP] = rot_mtx[:, :, 2]
            kv[:, :, i, ANG_VEL] = encoded_player[:, 12: 15]
            kv[:, :, i, BOOST] = encoded_player[:, 37]
            kv[:, :, i, DEMO] = encoded_player[:, 33]
            kv[:, :, i, ON_GROUND] = encoded_player[:, 34]
            kv[:, :, i, HAS_FLIP] = encoded_player[:, 36]

        kv[teams == 1] *= self._invert
        kv[np.argwhere(teams == 1), ..., (IS_MATE, IS_OPP)] = kv[
            np.argwhere(teams == 1), ..., (IS_OPP, IS_MATE)]  # Swap teams

        kv /= self._norm

        for i in range(n_players):
            q[i, :, 0, :kv.shape[-1]] = kv[i, :, i, :]

        self.convert_to_relative(q, kv)     # Relative forward with x and y
        # kv[:, :, :, 5:11] -= q[:, :, :, 5:11]

        # MASK players to true
        m[:, :, n_players: lim_players] = 1

        return [(q[i], kv[i], m[i]) for i in range(n_players)]

    def add_actions(self, obs: Any, previous_actions: np.ndarray, player_index=None) -> None:
        """
        Modify current obs to include action

        player_index=None means actions for all players should be provided

        Args:
            obs (Any): Current Observation
            previous_actions (np.ndarray)
            player_index (int): Player index identifying the respective agent to add the actions to
        """
        if player_index is None:
            for (q, kv, m), act in zip(obs, previous_actions):
                q[:, 0, ACTIONS] = act
        else:
            q, kv, m = obs[player_index]
            q[:, 0, ACTIONS] = previous_actions
