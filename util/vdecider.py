import numpy as np

decider = lambda p : 1.0 if p > 0.5 else 0.0    #activation function
vdecider = np.vectorize(decider)