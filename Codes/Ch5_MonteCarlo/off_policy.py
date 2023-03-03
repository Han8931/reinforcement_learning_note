import numpy as np
import operator
import gym

from foodtruck import FoodTruck
from policies import get_eps_greedy, get_random_policy
from policy_eval import base_policy, choose_action, policy_evaluation
from mc import get_trajectory
import pdb

act_key = lambda elem : elem[1]

def off_policy_mc(env, n_iter, eps, gamma):
    np.random.seed(0)
    states =  env.state_space
    actions = env.action_space
    Q = {s: {a: 0 for a in actions} for s in states}
    C = {s: {a: 0 for a in actions} for s in states}
    target_policy = {}
    behavior_policy = get_random_policy(states, actions)
    for i in range(n_iter):
        if i % 10000 == 0:
            print("Iteration:", i)
        trajectory = get_trajectory(env, behavior_policy)
        G = 0
        W = 1
        T = len(trajectory) - 1
        for t, sar in enumerate(reversed(trajectory)):
            s, a, r = sar
            G = r + gamma * G
            C[s][a] += W
            Q[s][a] += (W/C[s][a]) * (G - Q[s][a])
            a_best = max(Q[s].items(), key=act_key)[0]
            #a_best = max(Q[s].items(), key=operator.itemgetter(1))[0]
            target_policy[s] = a_best
            behavior_policy[s] = get_eps_greedy(actions, eps, a_best)

            if a != target_policy[s]:
                break

            W = W / behavior_policy[s][a]
    target_policy = {s: target_policy[s] for s in states if s in target_policy}
    return target_policy, Q

if __name__ == "__main__":
    foodtruck = FoodTruck()
    policy, Q = off_policy_mc(foodtruck, 300000, 0.05, 1)

    print(policy)
    print(Q)

