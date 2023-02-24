import numpy as np
import gym

from foodtruck import FoodTruck

import pdb

def base_policy(states):
    policy = {}
    for s in states:
        day, inventory = s
        prob_a = {} 
        if inventory >= 300:
            prob_a[0] = 1
        else:
            prob_a[200 - inventory] = 0.5
            prob_a[300 - inventory] = 0.5
        policy[s] = prob_a
    return policy
