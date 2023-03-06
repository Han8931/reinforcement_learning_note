import warnings ; warnings.filterwarnings('ignore')

import gym, gym_walk, gym_aima
import numpy as np
from pprint import pprint
from tqdm import tqdm_notebook as tqdm

from itertools import cycle

import random

np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)
