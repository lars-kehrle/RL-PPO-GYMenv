import gymnasium as gym
import torch
import numpy as np
from src.model.PPOAgent import PPOAgent
from src.networks.CartPoleAgentNetwork import CartPoleAgentNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
num_envs = 20

environment_name = 'CartPole-v1'
env = gym.make(environment_name)
envs = gym.make_vec(environment_name, num_envs=num_envs, vectorization_mode=gym.VectorizeMode.SYNC)

network = CartPoleAgentNetwork(env.observation_space.shape[0], env.action_space.n)
next_obs, _ = envs.reset()
next_dones = np.zeros(num_envs)

agent = PPOAgent(env.observation_space, env.action_space,
                 num_envs=num_envs,
                 agent_network_cls=CartPoleAgentNetwork,
                 device=device,
                 gamma=0.99,
                 learning_rate=0.003,
                 rollout_length=256,
                 nr_epochs=5,
                 batch_size=64,
                 use_gae=True,
                 gae_lambda=0.95,
                 clip_coef=0.2,
                 value_loss_coef=0.5,
                 entropy_loss_coef=0.01)

training_steps = 300
agent.train(envs, training_steps, env, 10, 10, False)
