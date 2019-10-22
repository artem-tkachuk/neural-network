'''
    Implementation of a simple neural network with 1 hidden layer
    Ⓒ Artem Tkachuk
'''

from training.train import train
from testing.test import test

def neural():

    datasets = [
        # {
        #     'name': 'netflix',
        #     'nTimes': 10,
        #     'rate': 0.0001
        # },
        {
            'name': 'SPECT',
            'nTimes': 1000,
            'rate': 0.001
        }
    ]

    mh = 13  # number of neurons in the hidden layer

    for ds in datasets:
        thetas = train(ds['name'], ds['nTimes'], ds['rate'], mh)
        test(ds['name'], thetas)