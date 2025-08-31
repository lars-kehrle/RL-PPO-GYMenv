import numpy as np
from src.networks.CartPoleAgentNetwork import CartPoleAgentNetwork
from src.model.PPOAgent import PPOAgent
import torch
import gymnasium as gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

environment_name = 'CartPole-v1'
env = gym.make(environment_name, render_mode="human")

network = CartPoleAgentNetwork(env.observation_space.shape[0], env.action_space.n)
network.load_actor()
network.to(device)
next_obs, _ = env.reset()
score = PPOAgent.evaluate(device, network, env, 1)
print(score)
