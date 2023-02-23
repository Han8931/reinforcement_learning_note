import numpy as np
import pdb

def get_P(m, p_up, p_down, p_left, p_right):
    "Define a transition probability matrix"
    m2 = m**2
    P = np.zeros((m2, m2))

    ix_map = {i + 1: (i // m, i % m) for i in range(m2)}
    for i in range(m2):
        for j in range(m2):
            r1, c1 = ix_map[i + 1]
            r2, c2 = ix_map[j + 1]
            rdiff = r1 - r2
            cdiff = c1 - c2
            if rdiff == 0:
                if cdiff == 1:
                    P[i, j] = p_left
                elif cdiff == -1:
                    P[i, j] = p_right
                elif cdiff == 0:
                    if r1 == 0:
                        P[i, j] += p_down
                    elif r1 == m - 1:
                        P[i, j] += p_up
                    if c1 == 0:
                        P[i, j] += p_left
                    elif c1 == m - 1:
                        P[i, j] += p_right
            elif rdiff == 1:
                if cdiff == 0:
                    P[i, j] = p_down
            elif rdiff == -1:
                if cdiff == 0:
                    P[i, j] = p_up
    return P

if __name__ == "__main__":
    m = 3 
    m2 = m**2 # N of grid

    # 3x3 grid, 
    # [6, 7, 8]: |(2,0), (2,1), (2,2)|
    # [3, 4, 5]: |(1,0), (1,1), (1,2)|
    # [0, 1, 2]: |(0,0), (0,1), (0,2)|
    q = np.zeros(m2) # Initial probability distribution
    q[m2//2] = 1 # A robot is at the center of the grid, (1,1).

    # 9x9 matrix (i,j), state i -> state j.
    robot_movement_p = [0.2, 0.3, 0.25, 0.25]
    P = get_P(3, *robot_movement_p)
    print(f"Robot Movement: {robot_movement_p}") 
    print(f"Transition probability matrix: \n{P}") 

    # Calculate the n-step transition probabilities. n=1 in this example
    n = 1
    Pn = np.linalg.matrix_power(P, n)
    n_transition = np.matmul(q, Pn)
    print(f"{n}-step transition : \n{n_transition}") 

    # A sample path in an ergodic Markov chain
    from scipy.stats import itemfreq

    s = 4 # First state
    n = 10**6
    visited = [s]

    for t in range(n):
        s = np.random.choice(m2, p=P[s,:])
        visited.append(s)

    # state = [0, 1, ..., 8]
    print(itemfreq(visited))







