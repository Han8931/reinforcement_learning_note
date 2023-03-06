import numpy as np
import gym

from foodtruck import FoodTruck
from policy_eval import expected_update, policy_evaluation

import pdb

def policy_improvement(env, v, s, actions, gamma):
    prob_a = {}
    if not env.is_terminal(s):
        max_q = np.NINF # Negative infinity
        best_a = None
        for a in actions:
            q_sa = expected_update(env, v, s, {a: 1}, gamma)
            if q_sa >= max_q:
                max_q = q_sa
                best_a = a
        prob_a[best_a] = 1
    else:
        # At the terminal state, q-value is always 0
        max_q = 0 
    return prob_a, max_q

def policy_iteration(env,  eps=0.1, gamma=1):
    np.random.seed(1)
    states = env.state_space
    actions = env.action_space
    policy = {s: {np.random.choice(actions): 1} for s in states}
    v = {s: 0 for s in states}
    while True:
        v = policy_evaluation(env, policy, v=v, eps=eps, gamma=gamma)
        old_policy = policy
        policy = {}
        for s in states:
            policy[s], _ = policy_improvement(env, v, s, actions, gamma)
        if old_policy == policy:
            break
    print("Optimal policy found!")
    return policy, v

def value_iteration(env, max_iter=100, eps=0.1, gamma=1):
    states = env.state_space
    actions = env.action_space
    v = {s: 0 for s in states}
    policy = {}
    k = 0
    while True:
        max_delta = 0
        for s in states:
            old_v = v[s]
            policy[s], v[s] = policy_improvement(env,
                                                 v,
                                                 s,
                                                 actions,
                                                 gamma)
            max_delta = max(max_delta, abs(v[s] - old_v))
        k += 1
        if max_delta < eps:
            print("Converged in", k, "iterations.")
            break
        elif k == max_iter:
            print("Terminating after", k, "iterations.")
            break
    return policy, v

if __name__=="__main__":
    foodtruck = FoodTruck()
    policy, v = policy_iteration(foodtruck)
    print(policy)
    print("Expected weekly profit:", v["Mon", 0])

    policy, v = value_iteration(foodtruck)
    print(policy)
    print("Expected weekly profit:", v["Mon", 0])

    
