import json
import os
import sys
from distutils.util import strtobool
import argparse

import configparser
from typing import Optional

import numpy
import numpy as np
import torch
from redis import Redis
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.reward_functions.common_rewards import ConstantReward
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_state_setters import construct_gamestates
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker, _unserialize, _serialize
from rocket_learn.utils.util import ExpandAdvancedObs
try:
    from rocket_learn.agent.pretrained_agents.human_agent import HumanAgent
except ImportError:
    pass

from training.aasma_body.obs import AasmaObsBuilder
from training.aasma_body.parser import AasmaAction
from training.aasma_body.reward import AasmaRewardFunction
from training.aasma_body.state import AasmaStateSetter, BetterRandom
from training.aasma_body.terminal import AasmaTerminalCondition, AasmaHumanTerminalCondition


def get_match(worker_counter_value: int, force_match_size: Optional[int],
              game_speed: int = 100, human_match: bool = False):
    """
    Setup agent match

    Args:
        worker_counter_value (int): Redis worker counter
        force_match_size (Optional[int]): Force game size from 1s to 3s
        game_speed (int): Game rollout velocity
        human_match (bool): Perform human match run

    Returns:
    Match
    """
    order = (2,)
    team_size = order[worker_counter_value % len(order)]

    if force_match_size:
        team_size = force_match_size

    terminals = AasmaTerminalCondition
    if human_match:
        terminals = AasmaHumanTerminalCondition

    return Match(
        reward_function=AasmaRewardFunction(),
        terminal_conditions=terminals(),
        obs_builder=AasmaObsBuilder(4),
        action_parser=AasmaAction(),
        #state_setter=AugmentSetter(AasmaStateSetter()),
        state_setter=AugmentSetter(BetterRandom()),
        self_play=True,
        team_size=team_size,
        game_speed=game_speed,
    )


def create_worker(host: str, name: str, password: str, worker_counter: str,
                  limit_threads: bool = True, send_game_states: bool = False, force_match_size: Optional[int] = None,
                  is_streamer: bool = False, human_match: bool = False):
    """
    Create RedisRolloutWorker to train agent

    Args:
        host (str): IP of the redis server
        name (str): Name of the user hosting the training
        password (str): Password of the redis server
        worker_counter (str): Redis Tag that tracks the number of current workers
        limit_threads (bool): Limit run to 1 thread
        send_game_states (bool):
        force_match_size (Optional[int]): Force game size from 1s to 3s
        is_streamer (bool): Perform stream run
        human_match (bool): Perform human match run

    Returns:
    RedisRolloutWorker
    """
    if limit_threads:
        torch.set_num_threads(1)
    redis = Redis(host=host, password=password)

    current_worker = redis.incr(worker_counter) - 1

    agents = None
    human = None

    if human_match:
        past_prob = 0
        eval_prob = 0
        game_speed = 1
        human = HumanAgent()

    else:
        past_prob = .2
        eval_prob = 0
        game_speed = 100

    return RedisRolloutWorker(redis, name,
                              match=get_match(current_worker, force_match_size,
                                              game_speed=game_speed,
                                              human_match=human_match),
                              past_version_prob=past_prob,
                              evaluation_prob=eval_prob,
                              send_gamestates=send_game_states,
                              streamer_mode=is_streamer,
                              pretrained_agents=agents,
                              human_agent=human,
                              sigma_target=0.5,
                              deterministic_old_prob=0.75)


def main():
    # Retrieve arguments
    assert len(sys.argv) >= 3

    parser = argparse.ArgumentParser(description='Launch Aasma worker')

    parser.add_argument('name', type=ascii,
                        help='<required> who is doing the work?')
    parser.add_argument('password', type=ascii,
                        help='<required> learner password')
    parser.add_argument('--compress', action='store_true',
                        help='compress sent data')
    parser.add_argument('--streamer_mode', action='store_true',
                        help='Start a streamer match, dont learn with this instance')
    parser.add_argument('--force_match_size', type=int, nargs='?', metavar='match_size',
                        help='Force a 1s, 2s, or 3s game')
    parser.add_argument('--human_match', action='store_true',
                        help='Play a human match against Aasma')

    args = parser.parse_args()

    name = args.name.replace("'", "")
    password = args.password.replace("'", "")
    compress = args.compress
    stream_state = args.streamer_mode
    force_match_size = args.force_match_size
    human_match = args.human_match

    if force_match_size is not None and (force_match_size < 1 or force_match_size > 3):
        parser.error("Match size must be between 1 and 3")

    # Parse values from config
    config_parser = configparser.ConfigParser(allow_no_value=True)
    config_parser.read("training/config.cfg")

    # - Reddis Settings
    ip = config_parser['Redis Configuration']['ip']
    worker_counter = config_parser['Redis Configuration']['worker_counter']

    # Run Worker
    try:
        worker = create_worker(ip, name, password, worker_counter,
                               limit_threads=True,
                               send_game_states=compress,
                               force_match_size=force_match_size,
                               is_streamer=stream_state,
                               human_match=human_match)
        worker.run()
    finally:
        print("Problem Detected. Killing Worker...")


if __name__ == '__main__':
    main()
