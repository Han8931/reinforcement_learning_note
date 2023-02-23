from grid import get_P
import numpy as np
import pdb

# Attaching rewards to the grid world example.
m2 = 9
P = np.zeros((m2+1, m2+1))
robot_movement_p = [0.2, 0.3, 0.25, 0.25]
P[:m2, :m2] = get_P(3, *robot_movement_p)
for i in range(m2):
    P[i, m2] = P[i, i]
    P[i, i] = 0

# The last row is the crashed state: cannot escape from this state
P[m2, m2] = 1

n = 10**5
avg_rewards = np.zeros(m2)
for s in range(9):
    for i in range(n):
        crashed = False
        s_next = s
        episode_reward = 0
        while not crashed:
            s_next = np.random.choice(m2+1, p=P[s_next, :])
            if s_next<m2:
                episode_reward+=1
            else:
                crashed=True
        avg_rewards[s]+=episode_reward
avg_rewards/=n

print(np.round(avg_rewards,2))
