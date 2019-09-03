from math import exp

# ended up not needing it
def sigmoid(x, derivative=False):
    sigm = 1.0 / (1 + exp(-x))
    return  sigm * (1.0 - sigm) if derivative else sigm