'''
    Implementation of a simple neural network with 1 hidden layer
    Ⓒ Artem Tkachuk
'''

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt;  plt.ion()
import numpy as np
import math
from scipy.special import expit
from util.readFile import readFile
from util.logLikelihood import logLikelihood
from util.sigmoid import sigmoid
from graph.graph import init, replot

def train(fn, nTimes, rate, mh):

    fileName = f'data/train/{fn}-train.txt'
    data, n, mx = readFile(fileName)

    features = data[:, :-1]              # matrix of training examples
    labels = data[:, -1]  # vector of corresponding labels
    y_hats = np.empty(labels.shape)

    thetas_y_hat = np.zeros((mh))         #parameters for input layer

    fig, ax, xdata, ydata, line = init(fn)     #plotting the log likelihood while training

    for k in range(nTimes):
        gradient_y_hat = np.zeros((mx))

        for example in range(n):
            # Forward Pass, computing hidden layer
            x, y = features[example], labels[example]

            #computing prediction
            sum = 0
            for i in range(mx):
                sum += thetas_y_hat[i] * x[i]
            y_hat = sigmoid(sum)
            y_hats[example] = y_hat
            delta = y - y_hat

            # computing gradients
            for i in range(mx):
                gradient_y_hat[i] += delta * x[i]

        # updating parameters
        for i in range(mx):
            thetas_y_hat[i] += rate * gradient_y_hat[i]


        LL = logLikelihood(labels, y_hats)
        print(LL)
        replot(fig, ax, line, nTimes, xdata, ydata, k, LL)


    plt.savefig(f'graph/pics/{fn}.png')

    return thetas_y_hat

# TODO loop for many h layers here? Or a matrix thing is possible with multiple layers too? Yes! 3d theta where i-th grid is current h
# TODO make it a vector of size n, if there are n hidden layers in the network
# TODO: reassure I underatand why does this work? (lines 2,3 of the body of the loop)
# TODO: Neural networks learns the best number of neurons and layers that give the best accuracy??