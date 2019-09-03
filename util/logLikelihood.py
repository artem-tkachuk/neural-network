import numpy as np

def logLikelihood(labels, sigm):
    return np.dot(labels, np.log(sigm)) + np.dot((1 - labels), np.log(1 - sigm))
