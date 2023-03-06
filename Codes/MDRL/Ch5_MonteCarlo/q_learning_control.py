import numpy as np
import operator
import gym

from foodtruck import FoodTruck
from policies import get_eps_greedy, get_random_policy
from policy_eval import base_policy, choose_action, policy_evaluation
from mc import get_trajectory
import pdb

act_key = lambda elem : elem[1]

def q_learning(env, gamma, eps, alpha, n_iter):
    np.random.seed(0)
    states =  env.state_space
    actions = env.action_space
    Q = {s: {a: 0 for a in actions} for s in states}
    policy = get_random_policy(states, actions)
    s = env.reset()
    for i in range(n_iter):
        if i % 100000 == 0:
            print("Iteration:", i)
        a_best = max(Q[s].items(),
                     key=act_key)[0]
        policy[s] = get_eps_greedy(actions, eps, a_best)
        a = choose_action(s, policy)
        s_next, reward, done, info = env.step(a)
        Q[s][a] += alpha * (reward
                            + gamma * max(Q[s_next].values())
                            - Q[s][a])
        if done:
            s = env.reset()
        else:
            s = s_next
    policy = {s: {max(policy[s].items(),
                 key=operator.itemgetter(1))[0]: 1}
                 for s in states}
    return policy, Q

if __name__ == "__main__":
    foodtruck = FoodTruck()
    policy, Q = q_learning(foodtruck, 1, 0.1, 0.01, 1000000)
    print(policy)

