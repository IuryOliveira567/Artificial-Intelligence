import random
import math

def argmaxall(gen):

    maxv = -math.inf
    maxvals = []

    for(e, v) in gen:
        if v > maxv:
            maxvals, maxv = [e], v
        elif v == maxv:
            maxvals.append(e)
    return maxvals

def argmaxe(gen):
    
    return random.choice(argmaxall(gen))

def argmax(lst):
    
    return argmaxe(enumerate(lst))

def argmaxd(dct):
    
    return argmaxe(dct.items())

def flip(prob):
    
    return random.random() < prob

def select_from_dist(item_prob_dist):
    
    ranreal = random.random()

    for(it, prob) in item_prob_dist.items():
        if ranreal < prob:
            return it
        else:
            ranreal -= prob

    raise RunTimeError(f"{item_prob_dist} is not a probability distribution")

def test():
    assert argmax([1, 6, 55, 3, 55, 23]) in [2, 4]
    print("Passed unit test in utilities")
    print("Run test_aipython() to test (almost) everything")













            
