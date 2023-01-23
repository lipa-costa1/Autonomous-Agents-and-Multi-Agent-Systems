from typing import Optional, Union

import numpy as np
import torch
from earl_pytorch import EARLPerceiver
from earl_pytorch.util.util import mlp
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.agent.policy import Policy
from torch import nn
from torch.nn import Linear, ReLU
from torch.nn.init import xavier_uniform_

from training.aasma_body.parser import AasmaAction


class ControlsPredictorDot(nn.Module):
    def __init__(self, in_features, features=32, layers=2, actions=None):
        super().__init__()
        if actions is None:
            self.actions = torch.from_numpy(AasmaAction.make_lookup_table()).float()
        else:
            self.actions = torch.from_numpy(actions).float()
        self.net = mlp(8, features, layers)
        self.emb_convertor = nn.Linear(in_features, features)

    def forward(self, player_emb: torch.Tensor, actions: Optional[torch.Tensor] = None):
        if actions is None:
            actions = self.actions
        player_emb = self.emb_convertor(player_emb)
        act_emb = self.net(actions.to(player_emb.device))

        if act_emb.ndim == 2:

            return torch.einsum("ad,bpd->bpa", act_emb, player_emb)

        return torch.einsum("bad,bpd->bpa", act_emb, player_emb)


class Aasma(nn.Module):
    """
    Wraps EARL (Extensible Attention-based Rocket League model) + an output and takes a single input
    """

    def __init__(self, earl, output):
        super().__init__()
        self.earl = earl
        self.relu = ReLU()
        self.output = output
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initiate parameters in the transformer model. Taken from PyTorch Transformer impl"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, input_arrays: tuple) -> Union[tuple, np.array]:
        """
        Performs forward propagation on the nn
        Args:
            input_arrays (tuple): Queries, Keys_and_Values, and Mask

        Returns:
            Results of the nn after forward propagation
        """
        q, kv, m = input_arrays
        res = self.earl(q, kv, m)
        weights = None
        if isinstance(res, tuple):
            res, weights = res
        res = self.output(self.relu(res))
        if isinstance(res, tuple):
            res = tuple(r[:, 0, :] for r in res)
        else:
            res = res[:, 0, :]
        if weights is None:
            return res
        return res, weights


def get_critic() -> nn.Module:
    """
    Builds Critic of the A3C System responsible for estimating the action-value (Q value) or state-value (V value)

    Notes:
        Q value (a, s) -> specifies how good it is for an agent to
            perform a particular action in a state with a policy π
        V value (s) -> specifies how good it is for the agent to
            be in a given state with a policy π
    Returns:
        NN Critic
    """
    return Aasma(EARLPerceiver(128, 2, 4, 1, query_features=32, key_value_features=24),
                 Linear(128, 1))


def get_actor() -> Policy:
    """
    Builds Actor of the A3C System responsible for
    updating the policy distribution in the direction suggested by the Critic.

    Returns:
        NN Actor
    """
    split = (90,)
    return DiscretePolicy(Aasma(EARLPerceiver(128, 2, 4, 1, query_features=32, key_value_features=24),
                                ControlsPredictorDot(128)), split)


def get_agent(actor_lr: float, critic_lr: float = None) -> ActorCriticAgent:
    """
    Builds Actor Critic Agent with both Actor and Critic NNs and the designed optimizer
    Args:
        actor_lr (float): actor learning rate
        critic_lr (float): critic learning rate

    Returns:
        ActorCriticAgent instance
    """
    actor = get_actor()
    critic = get_critic()
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": actor_lr},
        {"params": critic.parameters(), "lr": critic_lr if critic_lr is not None else actor_lr}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    return agent
