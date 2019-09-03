'''
    Implementation of a simple neural network with k hidden layers
    for CS109 @ Stanford
    Ⓒ Artem Tkachuk
'''

import numpy as np
from util.util import readFile, vdecider
from scipy.special import expit

def test(fn, thetas):

    fileName = f'datasets/test/{fn}-test.txt'
    tests, n, _ = readFile(fileName, logistic=True)

    features = tests[:, :-1]  # array of train examples
    labels = tests[:, -1]  # array of corresponding labels
    values = np.unique(labels) # all possible values the labels have

    p = expit(np.matmul(features, thetas.transpose()))
    y_hats = vdecider(p)

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

    of = open('results/neural.txt', 'a+')
    of.write(report)