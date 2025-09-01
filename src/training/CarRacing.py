import gymnasium as gym
import torch
import numpy as np
from src.model.PPOAgent import PPOAgent
from src.networks.CarRacingNetwork import CarRacingNetwork
import argparse
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
import os

parser = argparse.ArgumentParser(description="Train PPO on CarRacing")
parser.add_argument("--store_path", type=str, default=None,
                    help="Optional path to store checkpoints/models")

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_envs = 8

environment_name = 'CarRacing-v3'

eval_env = gym.make(environment_name)
eval_env = FrameStackObservation(GrayscaleObservation(eval_env), stack_size=4)

envs = gym.vector.SyncVectorEnv([
    lambda: FrameStackObservation(GrayscaleObservation(gym.make(environment_name)), stack_size=4)
    for _ in range(num_envs)
])

n_action = eval_env.action_space.shape[0]  # number of continuous actions
low = torch.tensor(eval_env.action_space.low, dtype=torch.float32)
high = torch.tensor(eval_env.action_space.high, dtype=torch.float32)

network = CarRacingNetwork(n_action,low,high,device)
network.to(device)

agent = PPOAgent(eval_env.observation_space, eval_env.action_space,
                 num_envs=num_envs,
                 agent_network_cls=network,
                 device=device,
                 gamma=0.99,
                 learning_rate=1e-4,
                 rollout_length=2048,
                 nr_epochs=6,
                 batch_size=256,
                 use_gae=True,
                 gae_lambda=0.95,
                 clip_coef=0.15,
                 value_loss_coef=0.5,
                 entropy_loss_coef=0.008,
                 store_path=args.store_path)

training_steps = 50000
agent.train(envs, training_steps, eval_env, 10, 10, False)
