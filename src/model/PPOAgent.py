import numpy as np
import torch
from torch import optim
import gymnasium as gym
from torch import nn

from src.networks.BaseAgentNetwork import BaseAgentNetwork


class PPOAgent:
    """
    Implement PPO training algorithm.
    """
    def __init__(self, observation_space, action_space,
                 num_envs: int,
                 agent_network_cls: BaseAgentNetwork,
                 device,
                 gamma: float,
                 learning_rate: float,
                 rollout_length: int,
                 nr_epochs: int,
                 batch_size: int,
                 use_gae: bool,
                 gae_lambda: float,
                 clip_coef: float,
                 value_loss_coef: float,
                 entropy_loss_coef: float,
                 store_path: str=None ):
        """
        Initialize the PPO algorithm and the parameters it uses.
        Args:
            observation_space: the (single) observation space of the environment.
            action_space: the (single) action space of the environment.
            num_envs: the number of (vectorized) environments.
            agent_network_cls: the class that implements the actor and critic networks.
            device: the device (cpu, cuda or mps) to use for training
            gamma: The discount factor.
            learning_rate: The learning rate.
            rollout_length: The lengths of the rollouts
            nr_epochs: The number of epochs to train for after each rollout.
            batch_size: The (mini-batch) size for training.
            use_gae: Use generalized advantage estimation (true) or n-step returns
            gae_lambda: The lambda parameter for generalized advantage estimation.
            clip_coef: The clipping coefficient of the PPO algorithm.
            value_loss_coef: The scaling of the value loss in the loss function.
            entropy_loss_coef: The scaling of the entropy loss in the loss function.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_network_cls = agent_network_cls
        self.num_envs = num_envs
        self.device = device

        # hyperparameters for training
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.rollout_length = rollout_length
        self.nr_epochs = nr_epochs
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.best_score = -np.inf

        # create the network
        self.agent_network = self.agent_network_cls
        self.agent_network.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.agent_network.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-5)

        # the rollout data will be kept as numpy arrays and just the mini batches will be moved to
        # tensors on the device.
        self.obs = np.zeros((self.rollout_length, self.num_envs) + self.observation_space.shape)
        self.actions = np.zeros((self.rollout_length, self.num_envs) + self.action_space.shape)
        self.log_probs = np.zeros((self.rollout_length, self.num_envs))
        self.rewards = np.zeros((self.rollout_length, self.num_envs))
        self.dones = np.zeros((self.rollout_length, self.num_envs))
        self.values = np.zeros((self.rollout_length, self.num_envs))

        # calculated returns and advantages
        self.returns = np.zeros((self.rollout_length, self.num_envs))
        self.advantages = np.zeros((self.rollout_length, self.num_envs))

        # keeping track of overall number of steps
        self.global_step = 0
        self.store_path = store_path

    def rollout(self, envs, next_obs, next_dones):
        """
        Calculate the rollout for the vectorized environment.

        Args:
            envs: The environments for the rollout. The number of envs must correspond to the
            number given in the constructor
            next_obs: The next observations for the rollout, i.e. the observations for the first step.
            next_dones: The next dones for the rollout.
        Returns: the observations and dones to be used to start the next rollout.

        """
        # do the rollouts for the number of steps
        for step in range(0, self.rollout_length):
            self.global_step += self.num_envs

            self.obs[step] = next_obs
            self.dones[step] = next_dones

            with torch.no_grad():
                next_obs_tensor = torch.tensor(next_obs).to(self.device)
                action, log_prob, value = self.agent_network.get_action_and_value(next_obs_tensor)
            self.values[step] = value.flatten()
            self.actions[step] = action
            self.log_probs[step] = log_prob

            next_obs, reward, next_dones, _, _ = envs.step(action)
            self.rewards[step] = reward

        # return the next obs and dones as they will be used for the next rollout
        return next_obs, next_dones


    def calculate_returns(self, next_obs, next_dones):
        """
        Calculate the returns and the advantages from the collected rollouts.
        Args:
            next_obs: the next observation at the end of the rollouts
            next dones: the next dones at the end of the rollouts
        """
        with torch.no_grad():
            # get the value for the next observation to bootstrap the returns
            next_obs_tensor = torch.tensor(next_obs).to(self.device)
            next_value = self.agent_network.get_value(next_obs_tensor).reshape(1, -1)

            # calculate the returns backwards from the rewards
            for t in reversed(range(self.rollout_length)):
                if t == self.rollout_length - 1:
                    next_is_non_terminal = 1.0 - next_dones
                    next_return = next_value
                else:
                    next_is_non_terminal = 1.0 - self.dones[t + 1]
                    next_return = self.returns[t + 1]
                # Wir greifen hier nur auf die Rewards und returns zu. (Der actor berechnet die beste policy, anhand dieser policy wurde gesampelt und wir erhaten dies sequence von rewards für die policy.
                #Daher sind die returns die actin-value function und die values ist die
                self.returns[t] = self.rewards[t] + next_is_non_terminal * next_return * self.gamma
                self.advantages[t] =  self.returns[t] - self.values[t]

    def calculate_gae(self, next_obs, next_dones):
        """
        Calculate the advantages from the rollouts using the GAE approach. This can be done
        iteratively from the end by multiplying the previous advantage by the gamma and lambda
        factors and adding the TD error.

        The returns can then be calculated from the advantages and the value functions.
        Args:
            next_obs: the next observation at the end of the rollouts
            next_dones: the next dones at the end of the rollouts
        """
        with (torch.no_grad()):
            # get the value for the next observation to bootstrap the returns
            next_obs_tensor = torch.tensor(next_obs).to(self.device)
            last_gae = 0
            for t in reversed(range(self.rollout_length)):
                if t == self.rollout_length - 1:
                    next_is_non_terminal = 1.0 - next_dones
                    next_value = self.agent_network.get_value(next_obs_tensor).reshape(1, -1)
                else:
                    next_is_non_terminal = 1.0 - self.dones[t + 1]
                    next_value = self.values[t + 1]
                # calculate TD error, advantage and returns. Save the alst GAE value in last_gae

                td_error = self.rewards[t] + self.gamma * next_value * next_is_non_terminal - self.values[t]
                last_gae = td_error + last_gae * self.gamma * self.gae_lambda * next_is_non_terminal
                self.returns[t] = last_gae + self.values[t]
                self.advantages[t] = last_gae

        return last_gae

    def train_epoch(self, verbose=False):
        """
        Train one epoch using the collected rollouts and calculated advantages. Looping over epochs needs to be done
        in the main training loop.
        """

        # we have 2D arrays of observations etc by step and environment, which we now reshape
        obs = self.obs.reshape((-1,) + self.observation_space.shape)
        actions = self.actions.reshape((-1,) + self.action_space.shape)
        log_probs = self.log_probs.reshape(-1)
        returns = self.returns.reshape(-1)
        advantages = self.advantages.reshape(-1)

        # we do shuffling of the indices and use the complete batch data
        indices = np.arange(self.rollout_length)
        np.random.shuffle(indices)

        # calculate the start and end positions of the minibatches
        for start in range(0, self.rollout_length, self.batch_size):
            # in case the rollout length is not a multiple of the batch size
            end = min(start + self.batch_size, self.rollout_length)
            mini_batch_indices = indices[start:end]

            # convert the minibatch data to tensors where needed:
            obs_tensor = torch.tensor(obs[mini_batch_indices], dtype=torch.float32).to(self.device)
            actions_tensor = torch.tensor(actions[mini_batch_indices]).to(self.device)
            advantages_tensor = torch.tensor(advantages[mini_batch_indices], dtype=torch.float32).to(self.device)
            returns_tensor = torch.tensor(returns[mini_batch_indices]).to(self.device)
            log_probs_tensor = torch.tensor(log_probs[mini_batch_indices]).to(self.device)

            # calculate the log probs and values using the current weights
            new_log_probs, entropy, new_value = self.agent_network.get_probs_and_value(obs_tensor, actions_tensor)

            # calculate the ratio between the old and new probabilities
            # YOUR CODE HERE
            ratio = torch.exp(new_log_probs - log_probs_tensor)
            # calculate the clipped PPO loss (negative, as we want to minimize)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
            clipped_loss = torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor)
            policy_loss = -clipped_loss.mean()

            # calculate the value loss and scaled value loss (multiply be value_loss_coef)
            # YOUR CODE HERE
            returns_tensor = returns_tensor.to(new_value.dtype)
            value_loss = nn.functional.mse_loss(new_value.view(-1), returns_tensor)
            v_loss_scaled = self.value_loss_coef * value_loss

            # calculate the entropy loss <-- used to do more exploration
            entropy_loss = -entropy.mean()
            entropy_loss_scaled = self.entropy_loss_coef * entropy_loss

            # use combined loss for gradient calculation
            loss = policy_loss + v_loss_scaled + entropy_loss_scaled

            self.optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

            # clipping is recommended but the actual clipping value is not so important
            nn.utils.clip_grad_norm_(self.agent_network.parameters(), 0.5)
            self.optimizer.step()

            # display some values if verbose, should best be done using wand or similar instead

            if verbose:
                print(f'p: {policy_loss:7.4} v: {v_loss_scaled:7.4} e:{entropy_loss_scaled:7.4}', end='\r')

    def train(self, envs: gym.Env, nr_steps: int, eval_env: gym.Env, eval_frequency: int,
              eval_episodes: int, verbose: bool = False):
        """
        Train the agent on the given environments for the given number of steps. One step is
        one rollout and a training batch for the number of epochs specified.
        Args:
            envs: The environments to train on. The number of environments must match the number
            given in the constructor parameter.
            nr_steps: The number of steps to train for.
            eval_env: The environment to evaluate on.
            eval_frequency: How often to run the evaluation.
            eval_episodes: How many episodes to run the evaluation.
            verbose: display training stats

        """
        next_obs, _ = envs.reset()
        next_dones = np.zeros(self.num_envs)

        for steps in range(nr_steps):
            next_obs, next_dones = self.rollout(envs, next_obs, next_dones)
            self.calculate_gae(next_obs, next_dones)
            for i in range(self.nr_epochs):
                self.train_epoch(verbose=verbose)
            if steps % eval_frequency == 0:
                evaluated_returns = self.evaluate(self.device, self.agent_network, eval_env, eval_episodes)
                if np.mean(evaluated_returns) >= self.best_score:
                    self.best_score = np.min(evaluated_returns)
                    self.agent_network.save_actor(self.store_path)
                if verbose:
                    print('')
                print(f'Step {steps} eval : Mean [{np.mean(evaluated_returns)}];  Min [{np.min(evaluated_returns)}]; Max [{np.max(evaluated_returns)}]')

    @staticmethod
    def evaluate(device, agent_network, env, nr_episodes: int):
        """
        Evaluate the agent on the given environment for the given number of episodes.
        Args:
            env: the environment to evaluate on.
            nr_episodes: the number of episodes to run the evaluation.

        Returns:
            the undiscounted returns
        """
        rewards = []
        for episode in range(nr_episodes):
            obs, _ = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            a = agent_network.get_deterministic_action(obs_tensor)
            done = False
            truncated = False
            episode_reward = 0
            while not done and not truncated:
                obs, reward, done, truncated, _ = env.step(a)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                a = agent_network.get_deterministic_action(obs_tensor)
                episode_reward += reward
            rewards.append(episode_reward)
        return rewards
