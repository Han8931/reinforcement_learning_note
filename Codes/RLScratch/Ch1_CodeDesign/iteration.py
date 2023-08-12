import itertools
from typing import Iterator
import pdb

def sqrt(a:float)->float:
    x = a/2
    while abs(x_n-x)>0.01:
        x_n = (x+a/x)/2
    return x_n

def sqrt(a:float)->Iterator[float]:
    x = a/2
    while True:
        x = (x+a/x)/2
        yield x

def converge(values: Iterator[float], threshold:float)->Iterator[float]:
    for a, b in itertools.pairwise(values):
        yield a

        if abs(a-b)<threshold:
            break


#def random_gen(y):
#    for i in range(y):
#        yield i
#        if i==3:
#            print(i)



if __name__ == "__main__":
#    for x in range(3):print(x)
#    elements = [1, 3, 2, 5]
#    print(list(itertools.takewhile(lambda x: x<5, elements)))

#    for i in random_gen(10):
#        print(f"I: {i}") 
    iterations = list(itertools.islice(sqrt(25), 10))
    print(iterations[-1])

    results = converge(sqrt(25), 0.0001)
    capped_results = itertools.islice(results, 100)

    for c in capped_results:
        print(c)


