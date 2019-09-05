import numpy as np
from math import sin

oftrain = open('train/sin.txt', 'w')
oftest = open('test/sin.txt', 'w')

test = 20000   #number
train = 10000  #of points

X = np.arange(train,step=0.01)
Y = np.arange(train, test, step = 0.01)

oftrain.write(f'1\n{train}\n')
for x in X:
    s = f'{x}: {sin(x)}\n'
    oftrain.write(s)

oftest.write(f'{train}\n{test}\n')
for y in Y:
    s = f'{y}: {sin(y)}\n'
    oftest.write(s)
