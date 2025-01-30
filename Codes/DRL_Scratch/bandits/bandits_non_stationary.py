import numpy as np

class NonStationaryBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms) 

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1*np.random.randn(self.arms) # Add some noise to the win rate
        if rate > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    def __init__(self, epsilon, alpha, action_size=10):
        self.epsilon = epsilon 
        self.Qs = np.zeros(action_size) # Q for each slot machine
        self.alpha = alpha

    def update(self, action, reward):
        """
        We can assign weight to the rewards.
        Q_n = Q_n-1 + alpha*(R_n-Q_n-1)
            = alpha*R_n + (1-alpha)*Q_n-1
            = alpha*R_n + alpha(1-alpha)*R_n-1+ ...
        This is called exponential moving average.
        """
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)

