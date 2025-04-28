import numpy as np
from collections import defaultdict

from numpy import argmax
from grid_world import GridWorld

def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states(): # Visit every state
        if state==env.goal_state:
            V[state]=0
            continue

        action_probs = pi[state]
        new_V = 0

        # Eval all actions at each state to update the value of the state
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V+=action_prob*(r+gamma*V[next_state]) # V_k+1 = \sum_a \pi [r+\gamma*V_k]
        V[state] = new_V

    return V

def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)
        delta = 0
        for state in env.states():
            t = abs(V[state]-old_V[state])
            if delta<t:
                delta = t
        if delta<threshold:
            break
    return V

env = GridWorld()
gamma = 0.9

pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
V = defaultdict(lambda: 0)
V = policy_eval(pi, V, env, gamma)

# for key, value in V.items():
#     print(f"{key}: {value:.2f}")

def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value==max_value:
            max_key = key
    return max_key

action_values = {0:0.1, 1:-0.3, 2:9.9, 3:-1.3}
max_action = max(action_values, key=action_values.get)
print(max_action)

def greedy_policy(V, env, gamma):
    pi = {}
    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r+gamma*V[next_state]
            action_values[action] = value
        max_action = argmax(action_values)
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi

def policy_iter(env, gamma, threshold=0.01):
    pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
    V = defaultdict(lambda: 0)
    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)
        if new_pi == pi:
            break
        pi = new_pi
    return pi

pi = policy_iter(env, gamma, threshold=0.01)
print(pi)

def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r+gamma*V[next_state]
            action_values.append(value)
        V[state] = max(action_values)
    return V






