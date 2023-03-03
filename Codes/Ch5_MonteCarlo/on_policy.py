import numpy as np
import operator
import gym

from foodtruck import FoodTruck
from policies import get_eps_greedy, get_random_policy
from policy_eval import base_policy, choose_action, policy_evaluation
from mc import get_trajectory
import pdb

act_key = lambda elem : elem[1]

def on_policy_first_visit_mc(env, n_iter, eps, gamma):
    np.random.seed(0)
    states =  env.state_space
    actions = env.action_space
    policy =  get_random_policy(states, actions)
    Q = {s: {a: 0 for a in actions} for s in states}
    Q_n = {s: {a: 0 for a in actions} for s in states}
    for i in range(n_iter):
        if i % 10000 == 0:
            print("Iteration:", i)
        trajectory = get_trajectory(env, policy)
        G = 0
        T = len(trajectory) - 1
        for t, sar in enumerate(reversed(trajectory)):
            s, a, r = sar
            G = r + gamma * G
            first_visit = True
            for j in range(T - t):
                s_j = trajectory[j][0]
                a_j = trajectory[j][1]
                if (s, a) == (s_j, a_j):
                    first_visit = False
            if first_visit:
                # I don't understand this part...
                Q[s][a] = Q_n[s][a] * Q[s][a] + G
                Q_n[s][a] += 1
                Q[s][a] /= Q_n[s][a]
                a_best = max(Q[s].items(), key=act_key)[0]
                policy[s] = get_eps_greedy(actions, eps, a_best)

    return policy, Q, Q_n

if __name__ == "__main__":
    foodtruck = FoodTruck()
    policy, Q, Q_n = on_policy_first_visit_mc(foodtruck, 300000, 0.05, 1)
    print(policy)
    print(Q)
    print(Q_n)

