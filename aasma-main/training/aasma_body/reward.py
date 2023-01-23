import numpy as np
from numpy import exp
from numpy.linalg import norm
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import CAR_MAX_SPEED
from rlgym.utils.gamestates import GameState, PlayerData

from ._constants import ORANGE_GOAL, BLUE_GOAL


class AasmaRewardFunction(RewardFunction):
    def __init__(
            self,
            team_spirit=1,  # 0.3 -> 0.5 -> 0.6
            opponent_punish_w=1
    ):
        self.team_spirit = team_spirit
        self.current_state = None
        self.last_state = None
        self.n = 0
        self.opponent_punish_w = opponent_punish_w
        self.state_quality = None
        self.rewards = None

    def _calculate_rewards(self, state: GameState) -> None:
        """
        Calculates rewards for the players

        - Flow:
            Calculate state based on ball and player positions
            Calculate either:
                - Goal Reward
                - Individual Player Rewards + Add state quality to players rewards
            Calculate reward based on Team Spirit and Opponent Punishment

        Args:
            state (GameState)
        """

        # Calculate rewards, positive for blue, negative for orange
        state_quality = self._state_qualities(state)
        player_rewards = np.zeros(len(state.players))
        mid = len(player_rewards) // 2

        # Handle goals with no scorer for critic consistency,
        # random state could send ball straight into goal
        d_blue = state.blue_score - self.last_state.blue_score
        d_orange = state.orange_score - self.last_state.orange_score

        # Either add goal reward (new episode started) or calculate other rewards
        if d_blue or d_orange:
            player_rewards[:mid], player_rewards[mid:] = self._goal_reward(d_blue, d_orange, state, mid)

        else:
            for i, player in enumerate(state.players):
                player_rewards[i] += self._calculate_player_rewards(i, player, state)

            player_rewards[:mid] += state_quality - self.state_quality
            player_rewards[mid:] -= state_quality - self.state_quality

        blue = player_rewards[:mid]
        orange = player_rewards[mid:]
        bm = np.nan_to_num(blue.mean())
        om = np.nan_to_num(orange.mean())

        # Team spirit reward calculations
        player_rewards[:mid] = ((1 - self.team_spirit) * blue + self.team_spirit * bm
                                - self.opponent_punish_w * om)
        player_rewards[mid:] = ((1 - self.team_spirit) * orange + self.team_spirit * om
                                - self.opponent_punish_w * bm)

        self.state_quality = state_quality
        self.last_state = state
        self.rewards = player_rewards

    def _state_qualities(self, state: GameState):
        """
        Get continuous rewards based on game state representing the state quality

        - Rewards
            State:
                Distance from ball to goal

        Args:
            state (GameState)

        Returns:
            Tuple [float, np.array]
                - state quality/2 (has it is given to both teams)
        """
        ball_pos = state.ball.position

        state_quality = (exp(-norm(ORANGE_GOAL - ball_pos) / CAR_MAX_SPEED)
                         - exp(-norm(BLUE_GOAL - ball_pos) / CAR_MAX_SPEED))

        # Half state quality because it is applied to both teams, thus doubling it in the reward distributing
        return state_quality / 2

    def _goal_reward(self, blue_goal_diff: int, orange_goal_diff: int, state: GameState, mid: int) -> tuple:
        """
        Calculate goal reward for both blue and orange teams in case of new goal occurrences

        Args:
            blue_goal_diff (int): New goals for the blue team
            orange_goal_diff (int): New goals for the orange team
            state (GameState)
            mid (int): Mid-value between players from blue team and orange team

        Returns:
            Tuple with both the goal rewards to the blue team and to the orange team
        """
        blue_reward = 0
        orange_reward = 0
        if blue_goal_diff > 0:
            orange_reward = -10
            blue_reward = 10
        if orange_goal_diff > 0:
            blue_reward = -10
            orange_reward = 10

        return blue_reward, orange_reward

    def _calculate_player_rewards(self, i: int, player: PlayerData, state: GameState) -> float:
        """
        Calculate individual player reward based on
            - Ball touched

        Args:
            i (int): Index of player in the players array
            player (PlayerData)
            state (GameState)

        Returns:
            Float with respective reward
        """
        if player.ball_touched:
            return self._ball_touched_reward(player, state)

        return 0

    def _ball_touched_reward(self, player: PlayerData, state: GameState) -> float:
        return 0.2

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Get reward for current state

        Args:
            player (PlayerData)
            state (GameState)
            previous_action (np.ndarray)

        Returns:
            Reward for current state
        """
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self._calculate_rewards(state)
            self.n = 0
        rew = self.rewards[self.n]
        self.n += 1
        return float(rew)  # / 3.2  # Divide to get std of expected reward to ~1 at start, helps value net a little

    def reset(self, initial_state: GameState) -> None:
        """
        Resets reward state

        Args:
            initial_state (GameState)
        """
        self.n = 0
        self.last_state = None
        self.rewards = None
        self.current_state = initial_state
        self.state_quality = self._state_qualities(initial_state)
