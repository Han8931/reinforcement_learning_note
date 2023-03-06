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

def estimate_state_values(P, m2, threshold):
    v = np.zeros(m2 + 1)
    max_change = threshold
    terminal_state = m2 
    while max_change >= threshold:
        max_change = 0
        for s in range(m2 + 1):
            v_new = 0
            for s_next in range(m2 + 1):
                r = 1 * (s_next != terminal_state)
                v_new += P[s, s_next] * (r + v[s_next])
            max_change = max(max_change, np.abs(v[s] - v_new))
            v[s] = v_new
    return np.round(v, 2)

print(estimate_state_values(P, 9, 0.005))
