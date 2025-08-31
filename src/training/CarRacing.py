import gymnasium as gym
import torch
import numpy as np
from src.model.PPOAgent import PPOAgent
from src.networks.CarRacingNetwork import CarRacingNetwork
import argparse
from gymnasium.wrappers import FrameStackObservation
import os

parser = argparse.ArgumentParser(description="Train PPO on CarRacing")
parser.add_argument("--store_path", type=str, default=None,
                    help="Optional path to store checkpoints/models")

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_envs = 1

environment_name = 'CarRacing-v3'
env = gym.make(environment_name)
env = FrameStackObservation(env, stack_size=4)

eval_env = gym.make(environment_name)
eval_env = FrameStackObservation(eval_env, stack_size=4)

n_action = env.action_space.shape[0]  # number of continuous actions
low = torch.tensor(env.action_space.low, dtype=torch.float32)
high = torch.tensor(env.action_space.high, dtype=torch.float32)

network = CarRacingNetwork(n_action,low,high,device)
network.to(device)
next_obs, _ = env.reset()
next_dones = np.zeros(num_envs)

agent = PPOAgent(env.observation_space, env.action_space,
                 num_envs=num_envs,
                 agent_network_cls=network,
                 device=device,
                 gamma=0.99,
                 learning_rate=2.5e-4,
                 rollout_length=256,
                 nr_epochs=4,
                 batch_size=64,
                 use_gae=True,
                 gae_lambda=0.95,
                 clip_coef=0.1,
                 value_loss_coef=0.5,
                 entropy_loss_coef=0.01,
                 store_path=args.store_path)

training_steps = 50000
agent.train(env, training_steps, eval_env, 20, 10, False)
