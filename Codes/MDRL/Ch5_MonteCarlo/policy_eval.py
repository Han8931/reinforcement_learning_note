import numpy as np
import gym

from foodtruck import FoodTruck

import pdb

def base_policy(states):
    """
    This is a base policy to be improved.
    """
    policy = {}
    for s in states:
        day, inventory = s
        prob_a = {} 

        # Replenish the inventory up to 200-300 at the begenning of a week day
        if inventory >= 300:
            # If the inventory >=300 then no action
            prob_a[0] = 1
        else:
            prob_a[200 - inventory] = 0.5
            prob_a[300 - inventory] = 0.5
        policy[s] = prob_a
    return policy

def expected_update(env, v, s, prob_a, gamma):
    expected_value = 0
    for a in prob_a:
        prob_next_s_r = env.get_transition_prob(s, a)
        for next_s, r in prob_next_s_r:
            # Value update (iteration)
            expected_value += prob_a[a] * prob_next_s_r[next_s, r] * (r + gamma * v[next_s])
    return expected_value

def policy_evaluation(env, policy, max_iter=100, v = None, eps=0.1, gamma=1):
    """
    - Evaluate a given policy
    - gamma=1 since this is an episodic task with a finite number of steps
    """
    if not v:
        # Init with zeros
        v = {s: 0 for s in env.state_space}
    k = 0
    while True:
        max_delta = 0
        for s in v:
            if not env.is_terminal(s):
                v_old = v[s]
                prob_a = policy[s]
                v[s] = expected_update(env, v, s, prob_a, gamma)
                max_delta = max(max_delta, abs(v[s] - v_old))
        k += 1
        if max_delta < eps:
            print(f"Converged in {k} iterations.")
            break
        elif k == max_iter:
            print(f"Terminating after {k} iterations.")
            break
    return v

def choose_action(state, policy):
    prob_a = policy[state]
    action = np.random.choice(a=list(prob_a.keys()), p=list(prob_a.values()))
    return action

def simulate_policy(policy, n_episodes):
    np.random.seed(0)
    foodtruck = FoodTruck()
    rewards = []
    for i_episode in range(n_episodes):
        state = foodtruck.reset()
        done = False
        ep_reward = 0
        while not done:
            action = choose_action(state, policy)
            state, reward, done, info = foodtruck.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    print("Expected weekly profit:", np.mean(rewards))

if __name__=="__main__":

    foodtruck = FoodTruck()
    policy = base_policy(foodtruck.state_space)

    # Evaluate profit we should expect in a week by following the base policy
    v = policy_evaluation(foodtruck, policy)
    print("Expected weekly profit:", v["Mon", 0])
    print(f"The state values: \n {v}")

    simulate_policy(policy, 1000)





