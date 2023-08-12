import random
import dataclasses

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Generic, TypeVar, Callable

import statistics
import numpy as np
import timeit
import pdb

T = TypeVar("T")

class Distribution(ABC, Generic[T]):
    @abstractmethod
    def sample(self)->T:
        pass

# Dataclass objects act like plain data
@dataclass
class Gaussian(Distribution[float]):
    def __init__(self, mean:float, std:float):
        self.mean = mean
        self.std = std

    def sample(self)->float:
        return np.random.normal(loc=self.mean, scale=self.std)

    def sample_n(self, n:int=100)->list:
        return np.random.normal(loc=self.mean, scale=self.std, size=n)

def expected_value(
        d: Distribution[T],
        f: Callable[[T], float],
        n: int
        )->float:
    return statistics.mean(f(d.sample()) for _ in range(n))

@dataclass(frozen=True)
class Coin(Distribution[str]):
    def sample(self):
        return "heads" if random.random() < 0.5 else "tails"

def payoff(coin: Coin)->float:
    return 1.0 if coin=="heads" else 0.0

if __name__=="__main__":

#    d = Gaussian(0, 1)
#    t1 = timeit.timeit(lambda: [d.sample() for _ in range(10)])
#    t2 = timeit.timeit(lambda: d.sample_n(10))
#    print(t1)
#    print(t2)
    
    coin_flip = Coin()
    mean = expected_value(coin_flip, payoff, 10)
    print(mean)

    f = lambda coin: 1.0 if coin=="heads" else 0.0
    mean = expected_value(coin_flip, f, 10)
    print(mean)








