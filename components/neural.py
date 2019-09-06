'''
    Implementation of a simple neural network with 1 hidden layer
    Ⓒ Artem Tkachuk
'''

from training.train import train
from testing.test import test

def neural():

    datasets = [
         # {
         #     'name': 'simple',
         #     'nTimes': 10000,
         #     'rate': 0.001
         # },
        # {
        #     'name': 'netflix',
        #     'nTimes': 50,
        #     'rate': 0.0001
        # },
        {
            'name': 'ancestry',
            'nTimes': 100,
            'rate': 0.0001
        },
    ]

    mh = 12  # number of neurons in the hidden layer

    for ds in datasets:
        thetas = train(ds['name'], ds['nTimes'], ds['rate'], mh)
        test(ds['name'], thetas)