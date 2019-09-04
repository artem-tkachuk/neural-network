import numpy as np

def logLikelihood(labels, y_hats):
    return np.sum(labels * np.log(y_hats) + (1 - labels) * np.log(1 - y_hats))
