from src.networks.BaseAgentNetwork import BaseAgentNetwork
from torch import nn


class LunarLanderAgentNetwork(BaseAgentNetwork):
    """
    Example implementation with specific actor/critic layers.
    """

    def __init__(self, n_obs, n_action):
        super().__init__(n_obs, n_action)
        self.actor = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_action)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.name = "LunarLander"
