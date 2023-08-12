import random
import dataclasses

from abc import ABC, abstractmethod
from dataclasses import dataclass

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

# Dataclass objects act like plain data
@dataclass(frozen=True)
class Die(Distribution):
    sides: int # Specify a data type, type annotations or type hints

    def sample(self):
        return random.randint(1, self.sides)

if __name__=="__main__":
    die = Die(6)
    #die.sides=10 # Give an error
    d10 = dataclasses.replace(die, sides=10)
    print(d10)

    # We can use frozen dataclass as a dictionary key
    sample_dict = {die: "abs"} 





