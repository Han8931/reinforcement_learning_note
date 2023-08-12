import random

from abc import ABC, abstractmethod

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

class Die(Distribution):
    def __init__(self, sides):
        self.sides = sides # Attribute

    # This one makes debugging much easier
    # A function cannot do this
    def __repr__(self):
        return f"Die(sides={self.sides})"

#    def __eq__(self, other):
#        return self.sides==other.sides

    def __eq__(self, other):
        if isinstance(other, Die):
            return self.sides==other.sides
        return False

    def sample(self):
        return random.randint(1, self.sides)




def roll_dice():
    return six_sided.sample()+six_sided.sample()

if __name__=="__main__":

    print(Die(6))
    six_sided = Die(6)
    print(six_sided==six_sided)

    # It shows False
    print(six_sided==Die(6))

