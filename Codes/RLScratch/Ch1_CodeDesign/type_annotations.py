## Dybanuc Typing

# Python is dynamically typed. It means that at the time of writing a program, 
# you generally don't know what type a variable (or other expression) will be

def double(x):
    # We don't know what type `x` will take.
    return x+x

# Type hint/annotation
# A tool like mypy and pyright can now check that you are not doing in a right way.
def double(x:int)->int:
    return x+x

# The typing module provides some more sophisticated type hints. 
# Any:  denotes the "wildcard type". 
# Union: you accept several types

from typing import Union
def print_thing(thing: Union[str, int]) -> None:
    if isinstance(thing, str):
        print("string", thing)
    else:
        print("number", thing)

# Optional[YourType] is just a shorthand for Union[YourType, None]. Union with None is very commonly used to indicate a potentially missing result.

## TypeVar

#This function accepts anything as the argument and returns it as is. How do you explain to the type checker that the return type is the same as the type of arg?

def identity(arg):
    return arg
from typing import TypeVar

T = TypeVar("T")
def identity(arg: T) -> T:
    return arg


def triple(string: Union[str, bytes]) -> Union[str, bytes]:
    return string * 3
print(triple("test"))

Anything = TypeVar("Anything")
def triple(string: Anything) -> Anything:
    return string * 3
print(triple("test"))


Anything = TypeVar("Anything", str, bytes)
def triple(string: Anything) -> Anything:
    return string * 3
print(triple("test"))


