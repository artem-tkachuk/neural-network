decider = lambda p : 1.0 if p > 0.5 else 0.0
vdecider = np.vectorize(decider)