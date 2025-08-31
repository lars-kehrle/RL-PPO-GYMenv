import gymnasium as gym
import torch
import numpy as np
from src.model.PPOAgent import PPOAgent
from src.networks.CarRacingNetwork import CarRacingNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
num_envs = 8

environment_name = 'CarRacing-v3'
env = gym.make(environment_name)
envs = gym.make_vec(environment_name, num_envs=num_envs, vectorization_mode=gym.VectorizeMode.SYNC)
n_action = env.action_space.shape[0]  # number of continuous actions
low = torch.tensor(env.action_space.low, dtype=torch.float32)
high = torch.tensor(env.action_space.high, dtype=torch.float32)

network = CarRacingNetwork(n_action,low,high,device)
network.to(device)
next_obs, _ = envs.reset()
next_dones = np.zeros(num_envs)

agent = PPOAgent(env.observation_space, env.action_space,
                 num_envs=num_envs,
                 agent_network_cls=network,
                 device=device,
                 gamma=0.99,
                 learning_rate=0.0005,
                 rollout_length=256,
                 nr_epochs=10,
                 batch_size=256,
                 use_gae=True,
                 gae_lambda=0.95,
                 clip_coef=0.2,
                 value_loss_coef=0.5,
                 entropy_loss_coef=0.01)

training_steps = 50000
agent.train(envs, training_steps, env, 50, 10, False)
