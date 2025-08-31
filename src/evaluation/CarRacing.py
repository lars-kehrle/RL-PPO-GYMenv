import numpy as np

from src.networks.CarRacingNetwork import CarRacingNetwork
from src.networks.CartPoleAgentNetwork import CartPoleAgentNetwork
from src.model.PPOAgent import PPOAgent
import torch
import gymnasium as gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

environment_name = 'CarRacing-v3'
env = gym.make(environment_name, render_mode="human")
print(env.action_space)
n_action = env.action_space.shape[0]  # number of continuous actions
low = torch.tensor(env.action_space.low, dtype=torch.float32)
high = torch.tensor(env.action_space.high, dtype=torch.float32)

network = CarRacingNetwork(n_action,low,high,device)
network.load_actor()
network.to(device)
next_obs, _ = env.reset()
score = PPOAgent.evaluate(device, network, env, 1)
print(score)
