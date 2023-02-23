"""
Reference:
- https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf
- https://gdmarmerola.github.io/ts-for-contextual-bandits/
"""
from contextual_bandits_user import UserGenerator

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

import pdb

class RegularizedLR(object):
    """
    alpha: control exploration and exploitation trade-off. smaller values -> exploit
    """
    def __init__(self, name, alpha, rlambda, n_dim):
        self.name = name

        # We can also increase incentives for exploration or exploitation 
        # by defining a hyperparameter α, which multiplies 
        # the variance of the Normal posteriors at prediction time
        self.alpha = alpha 

        self.rlambda = rlambda 
        self.n_dim = n_dim

        self.m = np.zeros(n_dim) # Mean 
        self.q = np.ones(n_dim) * rlambda
        self.w = self.get_sampled_weights()
        # We initialize all qi’s with a hyperparamenter λ, which is equivalent to the λ used in L2 regularization.

    def get_sampled_weights(self):
        """
        - For Thompson sampling
        - The weights are actually assumed to be distributed as independent gaussians:
        - W~N(m, q^-1)
        """
        w = np.random.normal(self.m, self.alpha * self.q**(-1/2))
        return w

    def loss(self, w, *args):
        X, y = args
        n = len(y)
        regularizer = 0.5 * np.dot(self.q, (w - self.m)**2)
        #np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j])))
        # np.log(1 + np.exp(np.dot(w, X[j]))) - y[j] * np.dot(w, X[j])

        # Logistic regression loss function
        # Note that this loss function is diff from the classification loss. 
        # We want to only maximize (or minimize) for a case of y=1
        pred_loss = sum([np.log(1 + np.exp(np.dot(w, X[j]))) - y[j] * np.dot(w, X[j]) for j in range(n)])
        return regularizer + pred_loss

    def fit(self, X, y):
        if y: # y==1
            X = np.array(X)
            y = np.array(y)

            minimization = minimize(self.loss,
                                    self.w,
                                    args=(X, y),
                                    method="L-BFGS-B",
                                    bounds=[(-10,10)]*3 + [(-1, 1)], # bounds on w
                                    options={'maxiter': 50})
            self.w = minimization.x
            self.m = self.w
            p = (1 + np.exp(-np.matmul(self.w, X.T)))**(-1)
            self.q = self.q + np.matmul(p * (1 - p), X**2)


    def calc_sigmoid(self, w, context):
        return 1 / (1 + np.exp(-np.dot(w, context)))

    def get_prediction(self, context):
        return self.calc_sigmoid(self.m, context)

    def sample_prediction(self, context):
        w = self.get_sampled_weights()
        return self.calc_sigmoid(w, context)

    def get_ucb(self, context):
        pred = self.calc_sigmoid(self.m, context)
        confidence = self.alpha * np.sqrt(np.sum(np.divide(np.array(context)**2, self.q)))
        ucb = pred + confidence
        return ucb

def calculate_regret(ug, context, ad_options, ad):
    action_values = {a: ug.logistic(ug.beta[a], context) for a in ad_options}
    best_action = max(action_values, key=action_values.get)
    regret = action_values[best_action] - action_values[ad] 
    return regret, best_action

def select_ad_eps_greedy(ad_models, context, eps):
    if np.random.uniform() < eps:
        return np.random.choice(list(ad_models.keys()))
    else:
        predictions = {ad: ad_models[ad].get_prediction(context)
                       for ad in ad_models}
        max_value = max(predictions.values());
        max_keys = [key for key, value in predictions.items() if value == max_value]
        return np.random.choice(max_keys)

def select_ad_ucb(ad_models, context):
    ucbs = {ad: ad_models[ad].get_ucb(context)
                   for ad in ad_models}
    max_value = max(ucbs.values());
    max_keys = [key for key, value in ucbs.items() if value == max_value]
    return np.random.choice(max_keys)

def select_ad_thompson(ad_models, context):
    samples = {ad: ad_models[ad].sample_prediction(context)
                   for ad in ad_models}
    max_value = max(samples.values());
    max_keys = [key for key, value in samples.items() if value == max_value]
    return np.random.choice(max_keys)

if __name__=="__main__":

    ug = UserGenerator()
    ad_options = ['A', 'B', 'C', 'D', 'E']
    exploration_data = {}
    data_columns = ['context', 'ad', 'click', 'best_action', 'regret', 'total_regret']
    exploration_strategies = ['eps-greedy', 'ucb', 'Thompson']

    for strategy in exploration_strategies:
        print("--- Init Model with", strategy)
        np.random.seed(0)

        # Create the LR models for each ad
        alpha, rlambda, n_dim = 0.5, 0.5, 4
        ad_models = {ad: RegularizedLR(ad, alpha, rlambda, n_dim) for ad in 'ABCDE'}

        # Initialize data structures
        X = {ad: [] for ad in ad_options}
        y = {ad: [] for ad in ad_options}

        results = []
        total_regret = 0
        # Start ad display
        for i in range(10**4):
            # User information
            context = ug.generate_user_with_context()
            if strategy == 'eps-greedy':
                eps = 0.1
                ad = select_ad_eps_greedy(ad_models, context, eps)
            elif strategy == 'ucb':
                ad = select_ad_ucb(ad_models, context)
            elif strategy == 'Thompson':
                ad = select_ad_thompson(ad_models, context)
            # Display the selected ad
            click = ug.display_ad(ad)
            # Store the outcome
            X[ad].append(context)
            y[ad].append(click)
            regret, best_action = calculate_regret(ug, context, ad_options, ad)
            total_regret += regret
            results.append((context, ad, click, best_action, regret, total_regret))
            # Update the models with the latest batch of data
            if (i + 1) % 500 == 0:
                print("Updating the models at i:", i + 1)
                for ad in ad_options:
                    ad_models[ad].fit(X[ad], y[ad])
                X = {ad: [] for ad in ad_options}
                y = {ad: [] for ad in ad_options}

        exploration_data[strategy] = {'models': ad_models, 'results': pd.DataFrame(results, columns=data_columns)}
    df_regret_comparisons = pd.DataFrame({s: exploration_data[s]['results'].total_regret
                                         for s in exploration_strategies})
    print(df_regret_comparisons)



