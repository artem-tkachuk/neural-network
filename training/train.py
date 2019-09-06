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

    thetas_h = np.ones((mx, mh))         #parameters for input layer
    thetas_y_hat = np.ones(mh)      #parameters for hidden layer

    fig, ax, xdata, ydata, line = init(fn)     #plotting the log likelihood while training

    for k in range(nTimes):
        gradient_h = np.zeros((mx, mh))
        gradient_y_hat = np.zeros((mh))

        for example in range(n):
            # Forward Pass, computing hidden layer
            x, y, h = features[example], labels[example], np.zeros((mh))
            for j in range(mh):
                sum = 0.0
                for i in range(mx):
                    sum += x[i] * thetas_h[i][j]
                h[j] = sigmoid(sum)

            #computing prediction
            sum = 0
            for j in range(mh):
                sum += thetas_y_hat[j] * h[j]
            y_hat = sigmoid(sum)
            y_hats[example] = y_hat

            # computing gradients
            for j in range(mh):
                gradient_y_hat[j] += (y - y_hat) * h[j]

            for i in range(mx):
                for j in range(mh):
                    gradient_h[i][j] += math.pow(thetas_y_hat[j], 2) * (1 - h[j]) * x[i]

        # updating parameters
        thetas_y_hat += rate * gradient_y_hat
        thetas_h += rate * gradient_h

        LL = logLikelihood(labels, y_hats)
        replot(fig, ax, line, nTimes, xdata, ydata, k, LL)


    plt.savefig(f'graph/pics/{fn}.png')

    return (thetas_h, thetas_y_hat, mh)

# TODO loop for many h layers here? Or a matrix thing is possible with multiple layers too? Yes! 3d theta where i-th grid is current h
# TODO make it a vector of size n, if there are n hidden layers in the network
# TODO: reassure I underatand why does this work? (lines 2,3 of the body of the loop)
# TODO: Neural networks learns the best number of neurons and layers that give the best accuracy??