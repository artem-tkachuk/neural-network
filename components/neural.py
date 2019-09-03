'''
    Implementation of a simple neural network with k hidden layers
    for CS109 @ Stanford
    Ⓒ Artem Tkachuk
'''

from train.neural import train
from test.neural import test

def neural():

    datasets = [
        {'name': 'simple', 'nTimes': 10000, 'rate': 0.0001},
        {'name': 'netflix', 'nTimes': 3000, 'rate': 0.0001},
    ]

    for ds in datasets:
        thetas = train(ds['name'], ds['nTimes'], ds['rate'])
        test(ds['name'], thetas)