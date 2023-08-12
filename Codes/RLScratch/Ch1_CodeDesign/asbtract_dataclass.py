import random

from abc import ABC, abstractmethod
from dataclasses import dataclass

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

@dataclass
class Die(Distribution):
    def __init__(self, sides:int)->None:
        self.sides = sides # Attribute
    
    def sample(self):
        return random.randint(1, self.sides)

if __name__=="__main__":

    print(Die(6))
#    six_sided = Die(6)
#    print(six_sided==six_sided)
#
#    print(six_sided==Die(6))
#    print(six_sided==None)

