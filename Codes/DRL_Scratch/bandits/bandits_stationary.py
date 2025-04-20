import numpy as np
import matplotlib.pyplot as plt

# Bandit problem 
class Bandit:
    def __init__(self, arms=10):
        """
        rates: win rate
        The win rates will not change over time. Thus, this is a stationary problem
        """
        self.rates = np.random.rand(arms) # Stationary distribution

    def play(self, arm):
        rate = self.rates[arm] 
        if rate > np.random.rand():
            return 1
        else:
            return 0


# This is a player
class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon # For exploration
        self.Qs = np.zeros(action_size) # Q for each slot machine
        self.ns = np.zeros(action_size) # # Trials for each machine

    def update(self, action, reward):
        self.ns[action] += 1

        # Incremental approach
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action] 

    def get_action(self):
        if np.random.rand() < self.epsilon: # Exploration
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)

steps = 1000
epsilon = 0.1

bandit = Bandit()
agent = Agent(epsilon)
total_reward = 0
total_rewards = []
rates = []

for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward

    total_rewards.append(total_reward)
    rates.append(total_reward / (step + 1))

print(total_reward)

plt.ylabel('Total reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()


# Due to the randomness introduced by the exploration, we should estimate the average return:

runs = 200
steps = 1000
epsilon = 0.1
all_rates = np.zeros((runs, steps))  # (2000, 1000)

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))

    all_rates[run] = rates

avg_rates = np.average(all_rates, axis=0)

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()

