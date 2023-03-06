"""
Consider a company that wants to advertise a product on various websites through digital banners, aiming to attract visitors to the product landing page. 
"""
import numpy as np

class BernoulliBandit(object):
    def __init__(self, p):
        self.p = p

    def display_ad(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward


if __name__ == "__main__":

    adA = BernoulliBandit(0.004)
    adB = BernoulliBandit(0.016)
    adC = BernoulliBandit(0.02)
    adD = BernoulliBandit(0.028)
    adE = BernoulliBandit(0.031)
    ads = [adA, adB, adC, adD, adE]

    n_test = 10000
    n_prod = 90000
    n_ads = len(ads) # 5
    Q = np.zeros(n_ads)  # Q, action values
    N = np.zeros(n_ads)  # N, total impressions
    total_reward = 0
    avg_rewards = []  # Save average rewards over time

    # A/B/n test
    for i in range(n_test):
        ad_chosen = np.random.randint(n_ads)
        R = ads[ad_chosen].display_ad()  # Observe reward
        N[ad_chosen] += 1
        Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen]) # Action value estimate
        total_reward += R
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)

    best_ad_index = np.argmax(Q)  # Find the best action
    print("The best performing ad is {}".format(chr(ord('A') + best_ad_index)))

    # Test in production
    ad_chosen = best_ad_index
    for i in range(n_prod):
        R = ads[ad_chosen].display_ad()
        total_reward += R
        avg_reward_so_far = total_reward / (n_test + i + 1)
        avg_rewards.append(avg_reward_so_far)

    import pandas as pd
    df_reward_comparison = pd.DataFrame(avg_rewards, columns=['A/B/n'])
    print(df_reward_comparison)

    # Epsilon Greedy
    eps = 0.1
    n_prod = 100000
    n_ads = len(ads)
    Q = np.zeros(n_ads)
    N = np.zeros(n_ads)
    total_reward = 0
    avg_rewards = []

    ad_chosen = np.random.randint(n_ads)
    for i in range(n_prod):
        R = ads[ad_chosen].display_ad()
        N[ad_chosen] += 1
        Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
        total_reward += R
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)

        # Select the next ad to display
        if np.random.uniform() <= eps:
            ad_chosen = np.random.randint(n_ads)
        else:
            ad_chosen = np.argmax(Q)

    df_reward_comparison['e-greedy: {}'.format(eps)] = avg_rewards

    # UCB: Upper confidence bounds 
    c = 0.1
    n_prod = 100000
    n_ads = len(ads)
    ad_indices = np.array(range(n_ads))
    Q = np.zeros(n_ads)
    N = np.zeros(n_ads)
    total_reward = 0
    avg_rewards = []

    for t in range(1, n_prod + 1):
        if any(N==0):
            ad_chosen = np.random.choice(ad_indices[N==0])
        else:
            uncertainty = np.sqrt(np.log(t) / N)
            UCB = Q +  c * uncertainty
            ad_chosen = np.argmax(UCB)

        R = ads[ad_chosen].display_ad()
        N[ad_chosen] += 1
        Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
        total_reward += R
        avg_reward_so_far = total_reward / t
        avg_rewards.append(avg_reward_so_far)

    df_reward_comparison['UCB, c={}'.format(c)] = avg_rewards

    # Thompson Sampling
    n_prod = 100000
    n_ads = len(ads)
    alphas = np.ones(n_ads)
    betas = np.ones(n_ads)
    total_reward = 0
    avg_rewards = []

    for i in range(n_prod):
        theta_samples = [np.random.beta(alphas[k], betas[k]) for k in range(n_ads)]
        ad_chosen = np.argmax(theta_samples)
        R = ads[ad_chosen].display_ad()
        alphas[ad_chosen] += R
        betas[ad_chosen] += 1 - R
        total_reward += R
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)
    df_reward_comparison['Thompson Sampling'] = avg_rewards



