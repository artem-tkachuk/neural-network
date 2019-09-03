'''
    Implementation of a simple neural network with k hidden layers
    for CS109 @ Stanford
    Ⓒ Artem Tkachuk
'''

import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt;  plt.ion()
from scipy.special import expit
from util.util import readFile, logLikelihood
from graph.graph import init, replot


def train(fn, nTimes, rate):
    fileName = f'datasets/train/{fn}-train.txt'
    data, n, m = readFile(fileName, logistic=True)

    features = data[:, :-1]  # array of train examples
    labels = data[:, -1]  # array of corresponding labels

    mh = m / 2  #number of neurons in the hidden layer  #TODO make it a vector of size n, if there are n hidden layers in the network

    thetas_h = np.zeros((mh, m))  # parameters   #TODO thetas h and thetas y_hat
    thetas_y_hat = np.zeros(1, mh)

    gradient = np.zeros(mh, m)  # gradients     # TODO ???

    fig, ax, xdata, ydata, line = init(fn)

    for k in range(nTimes):
        #loop for many h laters here? Or a matrix thing is possible with multiple layers too? Yes! 3d theta where i-th grid is current h
        h = expit(np.matmul(features, thetas.transpose()))
        y_hats = expit(np.matmul(h,thetas_y_hat.transpose()))

        diff = labels - y_hats

        # TODO chain rule here??
        # TODO matrix of partial derivatives?

        gradient = features * deriv[:, np.newaxis]
        thetas_h += np.sum(rate * gradient, axis=0)
        thetas_y_hat += np.sum(rate * gradient, axis=0)

        LL = logLikelihood(labels, y_hats)
        replot(fig, ax, line, nTimes, xdata, ydata, k, LL)

    plt.savefig(f'graphing/neural/{fn}.png')

    return thetas

# TODO: why does this work? (lines 2,3 of the body of the loop)
# TODO Neural networks learns the best number of neurons and layers that give the best accuracy??