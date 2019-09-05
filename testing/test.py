'''
    Implementation of a simple neural network with 1 hidden layer
    for CS109 @ Stanford
    Ⓒ Artem Tkachuk
'''

import numpy as np
from util.readFile import readFile
from util.vdecider import vdecider
from scipy.special import expit

def test(fn, thetas):

    fileName = f'data/test/{fn}-test.txt'
    tests, n, _ = readFile(fileName, logistic=True)

    features = tests[:, :-1]  # array of training examples
    labels = tests[:, -1]  # array of corresponding labels
    values = np.unique(labels) # all possible values the labels have

    thetas_h, thetas_y_hat = thetas
    print(thetas_h, thetas_y_hat)
    h = expit(np.matmul(features, thetas_h.transpose()))
    y_hats = expit(np.matmul(h, thetas_y_hat))

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