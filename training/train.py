'''
    Implementation of a simple neural network with 1 hidden layer
    Ⓒ Artem Tkachuk
'''

import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt;  plt.ion()
from scipy.special import expit
from util.readFile import readFile
from util.logLikelihood import logLikelihood
from graph.graph import init, replot


def train(fn, nTimes, rate):

    fileName = f'data/train/{fn}-train.txt'
    data, n, mx = readFile(fileName)
    mh = 12  # number of neurons in the hidden layer


    features = data[:, :-1]  # array of training examples
    labels = data[:, -1]  # array of corresponding labels
    labels = labels.reshape(labels.shape[0], 1)     #TODO remove

    thetas_h = np.ones((mx, mh))   #parameters for input layer
    thetas_y_hat = np.ones((mh,))  #parameters for hidden layer

    gradient_h = np.zeros((mx, mh))
    gradient_y_hat = np.zeros((mh,))

    fig, ax, xdata, ydata, line = init(fn)     #plotting the log likelihood while training

    for k in range(nTimes):
        #Forward Pass
        h = expit(np.matmul(features, thetas_h))
        y_hats = expit(np.matmul(h, thetas_y_hat))  #TODO reshape
        y_hats = y_hats.reshape(y_hats.shape[0], 1) #TODO remove

        # Backpropagation
        gradient_y_hat = np.sum(((labels - y_hats) * h), axis=0)

        deriv = (np.square(gradient_y_hat) * (np.sum((1 - h), axis=0))) #TODO remove
        feat = np.sum(features, axis=0) #TODO remove

        gradient_h = np.matmul(feat.reshape(feat.shape[0], 1), deriv.reshape(1, deriv.shape[0]))

        thetas_y_hat += rate * gradient_y_hat
        thetas_h += rate * gradient_h

        LL = logLikelihood(labels, y_hats)
        print(LL)   #TODO remove
        replot(fig, ax, line, nTimes, xdata, ydata, k, LL)

    plt.savefig(f'graph/pics/{fn}.png')

    return (thetas_h, thetas_y_hat)

# TODO loop for many h layers here? Or a matrix thing is possible with multiple layers too? Yes! 3d theta where i-th grid is current h
# TODO make it a vector of size n, if there are n hidden layers in the network
# TODO: reassure I underatand why does this work? (lines 2,3 of the body of the loop)
# TODO Neural networks learns the best number of neurons and layers that give the best accuracy??