import numpy as np
import operator
import gym

from foodtruck import FoodTruck
from policy_eval import base_policy, choose_action, policy_evaluation
import pdb

def get_eps_greedy(actions, eps, a_best):
    prob_a = {}
    n_a = len(actions)
    for a in actions:
        if a == a_best:
            prob_a[a] = 1 - eps + eps/n_a
        else:
            prob_a[a] = eps/n_a
    return prob_a

def get_random_policy(states, actions):
    policy = {}
    n_a = len(actions)
    for s in states:
        policy[s] = {a: 1/n_a for a in actions}
    return policy

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
                Q[s][a] = Q_n[s][a] * Q[s][a] + G
                Q_n[s][a] += 1
                Q[s][a] /= Q_n[s][a]
                a_best = max(Q[s].items(),
                             key=operator.itemgetter(1))[0]
                policy[s] = get_eps_greedy(actions, eps, a_best)

    return policy, Q, Q_n




if __name__ == "__main__":
    foodtruck = FoodTruck()
    policy, Q, Q_n = on_policy_first_visit_mc(foodtruck, 300000, 0.05, 1)

