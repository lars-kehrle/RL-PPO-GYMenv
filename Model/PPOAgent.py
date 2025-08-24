class AgentNetwork(nn.Module):
    """
    Build the agent networks.
    """
    def __init__(self, n_obs, n_action):
        super(AgentNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(n_obs, 32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32, n_action)
        )

    @staticmethod
    def init_linear(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x).numpy(force=True)

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.numpy(force=True)

    def get_action_and_value(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()

        return (action.numpy(force=True),
                probs.log_prob(action).numpy(force=True),
                self.critic(x).numpy(force=True))

    def get_probs_and_value(self, x, action):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.log_prob(action), probs.entropy(), self.critic(x)
