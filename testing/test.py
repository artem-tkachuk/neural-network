'''
    Implementation of a simple neural network with 1 hidden layer
    for CS109 @ Stanford
    Ⓒ Artem Tkachuk
'''

import numpy as np
from util.sigmoid import sigmoid
from util.readFile import readFile
from util.vdecider import vdecider
from scipy.special import expit

def test(fn, thetas):

    fileName = f'data/test/{fn}-test.txt'
    tests, n, mx = readFile(fileName, logistic=True)

    features = tests[:, :-1]  # array of training examples
    labels = tests[:, -1]  # array of corresponding labels
    values = np.unique(labels) # all possible values the labels have
    y_hats = np.empty(labels.shape)

    thetas_h, thetas_y_hat, mh = thetas

    for example in range(n):

        x, y, h = features[example], labels[example], np.zeros((mh))
        #computing h
        for j in range(mh):
            sum = 0.0
            for i in range(mx):
                sum += x[i] * thetas_h[i][j]
            h[j] = sigmoid(sum)

        # computing prediction
        sum = 0
        for j in range(mh):
            sum += thetas_y_hat[j] * h[j]
        y_hat = sigmoid(sum)
        y_hats[example] = y_hat

    y_hats = vdecider(y_hats)
    guessed = np.zeros(len(values), dtype=int) #number of guesses for each value
    total = np.zeros(len(values), dtype=int)   #quantity of each value

    #Testing
    for label, y_hat in zip(labels, y_hats):
        for i in range(len(values)):
            if values[i] == label:
                total[i] += 1
                if y_hat == label:
                    guessed[i] += 1
                break

    #Preparing report
    report = f'For "{fn}" dataset:\n'
    for i in range(len(values)):
        report += f'Class {values[i]}: tested {total[i]}, ' \
               f'correctly classified {guessed[i]}\n'
    report += f'Overall: tested {n}, correctly classified {guessed.sum()}\n'
    report += f'Accuracy: {float(guessed.sum()) / n}\n\n'

    of = open('results/results.txt', 'a+')
    of.write(report)