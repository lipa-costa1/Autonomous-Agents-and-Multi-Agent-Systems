import math
import os

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from torch.nn import Linear, Sequential, ReLU
from rocket_learn.utils.util import SplitLayer
from earl_pytorch import EARLPerceiver

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from training.agent import Aasma, ControlsPredictorDot


class Agent:
    def __init__(self):
        split = (90,)
        actor = DiscretePolicy(Aasma(EARLPerceiver(128, 2, 4, 1, query_features=32, key_value_features=24),
                                     ControlsPredictorDot(128)), split)

        cur_dir = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(cur_dir, "checkpoint.pt"), 'rb') as f:
            model = torch.load(f)

        #with open(os.path.join(cur_dir, "model.pt"), 'wb') as f:
        #    torch.save(model['actor_state_dict'], f)

        #with open(os.path.join(cur_dir, "model.pt"), 'rb') as f:
        actor.load_state_dict(model['actor_state_dict'])
        actor.eval()

        self.actor = actor

        torch.set_num_threads(1)
        self._lookup_table = self.make_lookup_table()
        self.state = None

    @staticmethod
    def make_lookup_table():
        actions = []

        # Ground Actions
        for throttle in (0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, 0])

        actions = np.array(actions)
        return actions

    def act(self, state, beta):
        state = tuple(torch.from_numpy(s).float() for s in state)

        with torch.no_grad():
            x = self.actor(state)
            if len(x) == 1:
                out, weights = x, None
            else:
                out, weights = x
        self.state = state

        out = (out,)
        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in out
            ],
            dim=1
        )

        # beta = 0.5
        if beta == 1:
            actions = np.argmax(logits, axis=-1)
        elif beta == -1:
            actions = np.argmin(logits, axis=-1)
        else:
            if beta == 0:
                logits[torch.isfinite(logits)] = 0
            else:
                logits *= math.log((beta + 1) / (1 - beta), 3)
            dist = Categorical(logits=logits)
            actions = dist.sample()

        # print(Categorical(logits=logits).sample())
        parsed = self._lookup_table[actions.numpy().item()]

        return parsed, weights