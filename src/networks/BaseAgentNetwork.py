import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from typing import ClassVar


class BaseAgentNetwork(nn.Module, ABC):
    """
    Abstract base class for agent networks.
    Subclasses should define self.actor and self.critic in __init__.
    """

    def __init__(self, n_obs: int, n_action: int):
        super(BaseAgentNetwork, self).__init__()
        self.n_obs = n_obs
        self.n_action = n_action
        self.name: ClassVar[str]
        self.actor: ClassVar[nn.Sequential]
        self.critic: ClassVar[nn.Sequential]

    @staticmethod
    def init_linear(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x: torch.Tensor):
        return self.critic(x).detach().cpu().numpy()

    def get_action(self, x: torch.Tensor):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.detach().cpu().numpy()

    def get_action_and_value(self, x: torch.Tensor):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return (
            action.detach().cpu().numpy(),
            probs.log_prob(action).detach().cpu().numpy(),
            self.critic(x).detach().cpu().numpy()
        )

    def get_probs_and_value(self, x: torch.Tensor, action: torch.Tensor):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return (
            probs.log_prob(action),
            probs.entropy(),
            self.critic(x)
        )

    def save_actor(self, path: str = None):
        if path is None:
            path = f"weights/{self.name}/ppo_actor.pth"
        torch.save(self.actor.state_dict(), path)

    def load_actor(self, path: str = None):
        if path is None:
            path = f"weights/{self.name}/ppo_actor.pth"
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def get_deterministic_action(self, x: torch.Tensor):
        logits = self.actor(x)
        action = torch.argmax(logits, dim=-1)  # greedy action
        return action.detach().cpu().numpy()
