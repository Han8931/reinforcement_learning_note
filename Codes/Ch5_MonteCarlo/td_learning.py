import numpy as np
import operator
import gym

from foodtruck import FoodTruck
from policies import get_eps_greedy, get_random_policy
from policy_eval import base_policy, choose_action, policy_evaluation
from mc import get_trajectory
import pdb

def one_step_td_prediction(env, policy, gamma, alpha, n_iter):
    np.random.seed(0)
    states = env.state_space
    v = {s: 0 for s in states}
    s = env.reset()
    for i in range(n_iter):
        a = choose_action(s, policy)
        s_next, reward, done, info = env.step(a)
        v[s] += alpha * (reward + gamma * v[s_next] - v[s])
        if done:
            s = env.reset()
        else:
            s = s_next
    return v

if __name__ == "__main__":
    foodtruck = FoodTruck()
    policy = base_policy(foodtruck.state_space)
    v = one_step_td_prediction(foodtruck, policy, 1, 0.01, 100000)
    print(v)

