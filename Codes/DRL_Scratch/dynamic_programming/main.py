from collections import defaultdict
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

for key, value in V.items():
    print(f"{key}: {value:.2f}")
