from random import randint

def six_sided():
    return randint(1,6)

def roll_dice():
    return six_sided()+six_sided()


