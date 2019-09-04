'''
    Implementation of a simple neural network with 1 hidden layer
    Ⓒ Artem Tkachuk
'''


import numpy as np
import matplotlib.pyplot as plt


def init(fn):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.set_ylim(-1.0, 1.0)

    plt.xlabel('n Times')
    plt.ylabel('LL(Θ)')
    plt.title(f'"{fn}"dataset')

    xdata, ydata = [], []
    line, = ax.plot(xdata, ydata, 'r-')

    return (fig, ax, xdata, ydata, line)


def replot(fig, ax, line, nTimes, xdata, ydata, k, LL):
    frequency = 25
    _, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if k > xmax:
        xmax = nTimes if 12 * k > nTimes else 12 * k
    if LL > ymax:
        ymax = 1.2 * LL
    if LL < ymin:
        ymin = 1.2 * LL
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)

    plt.xticks(np.arange(0, xmax, nTimes / 10))
    plt.yticks(np.arange(ymin, ymax, (ymax - ymin) / 10))

    xdata.append(k + 1)
    ydata.append(LL)
    line.set_data(xdata, ydata)


    if k % (nTimes / frequency) == 0:
        fig.canvas.draw()
        fig.canvas.flush_events()

