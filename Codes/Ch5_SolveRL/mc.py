import numpy as np
import gym

from foodtruck import FoodTruck
from policy_eval import base_policy, choose_action, policy_evaluation
import pdb

def first_visit_return(returns, trajectory, gamma):
    G = 0
    T = len(trajectory) - 1
    for t, sar in enumerate(reversed(trajectory)):
        s, a, r = sar # (s,a,r) tuple
        G = r + gamma * G
        first_visit = True
        for j in range(T - t):
            if s == trajectory[j][0]:
                first_visit = False
        if first_visit:
            if s in returns:
                returns[s].append(G)
            else:
                returns[s] = [G]
    return returns

def get_trajectory(env, policy):
    trajectory = []
    state = env.reset()
    done = False
    sar = [state]
    while not done:
        action = choose_action(state, policy) # (randomly) pick an action based on policy
        state, reward, done, info = env.step(action)
        sar.append(action)
        sar.append(reward)

        trajectory.append(sar)

        sar = [state]

    return trajectory

def first_visit_mc(env, policy, gamma, n_trajectories):
    np.random.seed(0)
    returns = {}
    v = {}
    for i in range(n_trajectories):
        trajectory = get_trajectory(env, policy)
        returns = first_visit_return(returns, trajectory, gamma)

    for s in env.state_space:
        if s in returns:
            v[s] = np.round(np.mean(returns[s]), 1)
    return v

if __name__ == "__main__":
    foodtruck = FoodTruck()
    policy = base_policy(foodtruck.state_space)

#    v_est = first_visit_mc(foodtruck, policy, 1, 10000)
#    print(v_est)
#
#    v_true = policy_evaluation(foodtruck, policy)
#    print(v_true)


#    v_est = first_visit_mc(foodtruck, policy, 1, 5)
#    print({s: v_est[s] for s in sorted(v_est)})
#
#    v_est = first_visit_mc(foodtruck, policy, 1, 100)
#    print({s: v_est[s] for s in sorted(v_est)})
#
#    v_est = first_visit_mc(foodtruck, policy, 1, 1000)
#    print({s: v_est[s] for s in sorted(v_est)})





