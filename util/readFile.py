import numpy as np
import io

def readFile(fileName, mode='r', logistic = False):
    fileData = open(fileName, mode)
    m = int(fileData.readline()) + 1  # number of features + theta0
    n = int(fileData.readline())      # number of training examples
    lines = fileData.read()
    lines = lines.replace(":", "")
    data = np.genfromtxt(io.StringIO(lines))
    data = np.insert(data, 0, 1, axis=1)
    return (data, n, m)