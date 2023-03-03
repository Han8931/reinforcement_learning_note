
def get_eps_greedy(actions, eps, a_best):
    """Build epsilon-greedy policy"""
    prob_a = {}
    n_a = len(actions)
    for a in actions:
        if a == a_best:
            prob_a[a] = 1 - eps + eps/n_a
        else:
            prob_a[a] = eps/n_a
    return prob_a

def get_random_policy(states, actions):
    """
    A random poilcy where all actions are equally likely to be taken.
    """
    policy = {}
    n_a = len(actions)
    for s in states:
        policy[s] = {a: 1/n_a for a in actions}
    return policy
