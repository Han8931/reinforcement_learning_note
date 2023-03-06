import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

#import plotly.offline
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go
#import cufflinks as cf

class UserGenerator(object):
    def __init__(self):
        self.beta = {}
        # b1, b2, b3, b4/ const, device, location, age
        # True beta parameters
        # Different ads have their own user interactions
        self.beta['A'] = np.array([-4, -0.1, -3, 0.1])
        self.beta['B'] = np.array([-6, -0.1, 1, 0.1])
        self.beta['C'] = np.array([2, 0.1, 1, -0.1])
        self.beta['D'] = np.array([4, 0.1, -3, -0.2])
        self.beta['E'] = np.array([-0.1, 0, 0.5, -0.01])
        self.context = None

    def logistic(self, beta, context):
        f = np.dot(beta, context)
        p = 1 / (1 + np.exp(-f)) # prob of user CTR
        return p

    def display_ad(self, ad):
        if ad in ['A', 'B', 'C', 'D', 'E']:
            p = self.logistic(self.beta[ad], self.context)
            reward = np.random.binomial(n=1, p=p)
            return reward
        else:
            raise Exception('Unknown ad!')

    def generate_user_with_context(self):
        """
        - Generate random users
        - No assumptions about any correlations between attributes
        """
        # 0: International, 1: U.S.
        location = np.random.binomial(n=1, p=0.6)

        # 0: Desktop, 1: Mobile
        device = np.random.binomial(n=1, p=0.8)

        # User age changes between 10 and 70,
        # with mean age 34
        age = 10 + int(np.random.beta(2, 3) * 60)
        # Add 1 to the concept for the intercept
        self.context = [1, device, location, age]
        return self.context

if __name__ == "__main__":
    test = UserGenerator()
    print(test)
