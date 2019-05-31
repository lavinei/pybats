import numpy as np


def MSE(y, f):
    return np.mean((y - f)**2)


def MAD(y, f):
    return np.mean(np.abs(y-f))


def WAPE(y, f):
    return np.sum(np.abs(y-f)) / np.sum(y)


def WAFE(y, f):
    return np.sum(np.abs(y-f)) / ((np.sum(y) + np.sum(f))/2)


def ZAPE(y, f):
    nonzeros = y.nonzero()[0]
    n = len(y)
    loss = np.copy(f)
    loss[nonzeros] = np.abs(y[nonzeros] - f[nonzeros]) / y[nonzeros]
    return np.mean(loss)


def scaledMSE(y, f, ymean = None):
    if ymean is None:
        # First check if the 'y' vector is longer than f
        ny = len(y)
        nf = len(f)
        ymean = np.cumsum(y) / np.arange(1, ny+1)
        # Assume that the forecasts and y vector terminate at the same point
        y = y[-nf:]
        ymean = ymean[-nf:]
    return np.mean(((y.reshape(-1) - f.reshape(-1)) ** 2 / (ymean.reshape(-1) ** 2)))
