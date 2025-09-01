import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

from src.networks.BaseAgentNetwork import BaseAgentNetwork


class CarRacingNetwork(nn.Module):
    """
    Example implementation with specific actor/critic layers.
    """

    def __init__(self, n_action, low, high, device):
        super().__init__()
        self.low = low.to(device)
        self.high = high.to(device)
        self.image_encoder = nn.Sequential(
            #96 96 3
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.Dropout(0.2),
            nn.ReLU())

        self.actor_mean = nn.Sequential(
            nn.Linear(512, 256),
            torch.nn.ReLU(),
            nn.Linear(256, 256),
            torch.nn.ReLU(),
            nn.Linear(256, n_action)
        )

        self.log_std = nn.Parameter(torch.ones(n_action) * np.log(1), requires_grad=False)
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            torch.nn.ReLU(),
            nn.Linear(256, 256),
            torch.nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.name = "CartRacing"

    def preprocess_obs(self, obs):
        obs = obs.float() / 255.0
        return obs

    def actor_forward(self, obs):
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        x = self.preprocess_obs(obs)
        assert torch.isfinite(x).all(), "Input to image_encoder contains NaN or Inf"
        assert not torch.isnan(x).any(), "Input to image_encoder contains NaN or Inf"
        x = self.image_encoder(x)
        mean = self.actor_mean(x).squeeze()
        log_std = self.log_std
        return mean, log_std

    def critic_forward(self, obs):
        x = self.preprocess_obs(obs)
        x = self.image_encoder(x)
        return self.critic(x)

    @staticmethod
    def init_linear(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x: torch.Tensor):
        return self.critic_forward(x).detach().cpu().numpy()

    def get_action_and_value(self, obs):
        mean, log_std = self.actor_forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Sample action with reparameterization trick
        raw_action = dist.rsample()

        # Apply tanh to map to [-1, 1]
        action = torch.tanh(raw_action)

        # Rescale to environment bounds
        action_scaled = self.low + (action + 1.0) * 0.5 * (self.high - self.low)

        # Correct log-prob for tanh squashing
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)

        value = self.critic_forward(obs).view(-1)
        return action_scaled.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), value.detach().cpu().numpy()

    def get_probs_and_value(self, obs, action_tensor):
        mean, log_std = self.actor_forward(obs)
        std = log_std.exp().clamp(min=1e-6)
        dist = Normal(mean, std)
        # invert scaling first

        action_in_tanh = 2 * (action_tensor - self.low) / (self.high - self.low) - 1.0
        eps = 1e-6
        action_in_tanh = torch.clamp(action_in_tanh, -1 + eps, 1 - eps)
        # compute log-prob in raw space
        log_prob_raw = dist.log_prob(torch.atanh(action_in_tanh)).sum(-1)
        # apply tanh correction
        log_prob_corrected = log_prob_raw - torch.log(1 - action_in_tanh.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1, keepdim=True)
        new_value = self.critic_forward(obs).view(-1)
        return log_prob_corrected, entropy, new_value

    def get_action(self, x: torch.Tensor):
        mean, log_std = self.actor_forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Sample action with reparameterization trick
        raw_action = dist.rsample()

        # Apply tanh to map to [-1, 1]
        action = torch.tanh(raw_action)

        action_scaled = (self.low + (action + 1.0) * 0.5 * (self.high - self.low))
        action_scaled = action_scaled.squeeze(0)
        # Rescale to environment bounds

        return action_scaled.detach().cpu().numpy()

    def save_actor(self, path: str = None):
        if path is None:
            path = f"weights/{self.name}/ppo_actor.pth"
        # save the encoder + actor MLP + mean + logstd
        torch.save({
            'image_encoder': self.image_encoder.state_dict(),
            'actor_mean': self.actor_mean.state_dict(),
            'log_std': self.log_std,
            'critic': self.critic.state_dict(),
        }, path)

    def load_actor(self, path: str = None):
        if path is None:
            path = f"weights/{self.name}/ppo_actor.pth"
        checkpoint = torch.load(path)
        self.image_encoder.load_state_dict(checkpoint['image_encoder'])
        self.actor_mean.load_state_dict(checkpoint['actor_mean'])
        self.log_std = checkpoint['log_std']
        self.eval()

    def get_deterministic_action(self, x: torch.Tensor):
        self.eval()
        mean, _ = self.actor_forward(x)
        action = torch.tanh(mean)  # squash mean
        action_scaled = self.low + (action + 1.0) * 0.5 * (self.high - self.low)
        # detach, convert, and fix shape/dtype
        return action_scaled.squeeze(0).detach().cpu().numpy()

    def load_actor_and_critic(self, path: str = None):
        if path is None:
            path = f"weights/{self.name}/ppo_actor.pth"
        checkpoint = torch.load(path)
        self.image_encoder.load_state_dict(checkpoint['image_encoder'])
        self.actor_mean.load_state_dict(checkpoint['actor_mean'])
        self.actor_logstd = checkpoint['actor_logstd']
        self.critic.load_state_dict(checkpoint['critic'])
