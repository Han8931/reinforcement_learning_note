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

R = np.ones(m2+1)
R[-1] = 0

inv = np.linalg.inv(np.eye(m2+1)-0.9999*P)
v = np.matmul(inv, np.matmul(P,R))
print(np.round(v,2))
