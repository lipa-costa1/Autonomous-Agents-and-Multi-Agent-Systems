from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

AasmaActionOLD = KBMAction


class AasmaAction(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        """
        Defines possible actions

            throttle(int) -> [0, 1]
            steer(int) (or yaw) -> [-1, 1]
            boost(bool) -> [0, 1]

        - Conditions
            if boost == 1
                -> throttle = 1 (accelerates while boosting)

        - Array format (one action)
            [throttle (or boost), steer (or yaw), pitch ->0, yaw (or steer), roll ->0, jump ->0, boost, handbrake ->0]

        Returns:
            Numpy array with possible actions
        """
        actions = []

        # Ground Actions
        for throttle in (0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, 0])

        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        """
        Retrieves action space

        Returns:
            Discrete action space with size of possible actions
        """
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        """
        Pass through that allows both multiple types of agent actions while still parsing Aasma

        Strip out fillers, pass through 8sets, get look up table values, recombine

        Args:
            actions (Any): Array of actions
            state (GameState): Not used

        Returns:
            Numpy ndarray with parsed actions
        """
        parsed_actions = []
        for action in actions:

            if action.size != 8:    # Reconstruct
                if action.shape == 0:   # Guarantee at least one dimension
                    action = np.expand_dims(action, axis=0)
                # To allow different action spaces, pad out short ones (assume later unpadding in parser)
                action = np.pad(action.astype('float64'), (0, 8 - action.size), 'constant', constant_values=np.NAN)

            if np.isnan(action).any():  # Detect padding -> delete to go back to original
                stripped_action = (action[~np.isnan(action)]).squeeze().astype('int')
                parsed_actions.append(self._lookup_table[stripped_action])
            else:
                parsed_actions.append(action)

        return np.asarray(parsed_actions)
