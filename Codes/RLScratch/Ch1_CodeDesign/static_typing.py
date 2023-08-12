import random
import dataclasses

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Generic, TypeVar

import statistics

T = TypeVar("T")

class Distribution(ABC, Generic[T]):
    @abstractmethod
    def sample(self)->T:
        pass

# Dataclass objects act like plain data
@dataclass(frozen=True)
class Die(Distribution[int]):
    sides: int # Specify a data type, type annotations or type hints

    def sample(self):
        return random.randint(1, self.sides)

def expected_value(d: Distribution[int], n:int = 100)->float:
    return statistics.mean(d.sample() for _ in range(n))

if __name__=="__main__":

    print(expected_value(Die(6), 100))






