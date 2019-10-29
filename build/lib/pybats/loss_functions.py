import numpy as np


def MSE(y, f):
    """
    Mean squared error (MSE).

    .. math:: MSE(y, f) = (1/n) * \sum_{i=1:n} (y_i-f_i)^2

    The optimal point forecast to minimize the MSE is the mean.

    .. code::

        k = 1
        MSE(y[forecast_start + k - 1:forecast_end + k], mean(samples))

    :param y: Observation vector
    :param f: Point forecast vector
    :return: Mean squared error (MSE)
    """

    y = np.ravel(y)
    f = np.ravel(f)
    return np.mean((y - f)**2)


def MAD(y, f):
    """
    Mean absolute deviation (MAD).

    .. math:: MAD(y, f) = (1/n) * \sum_{i=1:n} |y_i-f_i|

    The optimal point forecast to minimize the MAD is the median.

    .. code::

        k = 1
        MAD(y[forecast_start + k - 1:forecast_end + k], median(samples))

    :param y: Observation vector
    :param f: Point forecast vector
    :return: Mean absolute deviation (MAD)
    """

    y = np.ravel(y)
    f = np.ravel(f)
    return np.mean(np.abs(y-f))


def MAPE(y, f):
    """
    Mean absolute percent error (MAPE).

    .. math:: MAPE(y, f) = (1/n) * \sum_{i=1:n} |y_i-f_i| / y_i

    The optimal point forecast to minimize the MAPE is the (-1)-median. However, it is common to use the median point forecast, which is similar.

    .. code::

        k = 1
        MAPE(y[forecast_start + k - 1:forecast_end + k], m_one_median(samples))

    :param y: Observation vector
    :param f: Point forecast vector
    :return: Mean absolute percent error (MAPE)
    """

    y = np.ravel(y)
    f = np.ravel(f)
    return 100*np.mean(np.abs((y - f)) / y)


def WAPE(y, f):
    y = np.ravel(y)
    f = np.ravel(f)
    return 100*np.sum(np.abs(y-f)) / np.sum(y)


def WAFE(y, f):
    y = np.ravel(y)
    f = np.ravel(f)
    return 100*np.sum(np.abs(y-f)) / ((np.sum(y) + np.sum(f))/2)


def ZAPE(y, f):
    y = np.ravel(y)
    f = np.ravel(f)
    nonzeros = y.nonzero()[0]
    n = len(y)
    loss = np.copy(f)
    loss[nonzeros] = np.abs(y[nonzeros] - f[nonzeros]) / y[nonzeros]
    return 100*np.mean(loss)


def scaledMSE(y, f, ymean = None):
    y = np.ravel(y)
    f = np.ravel(f)
    if ymean is None:
        # First check if the 'y' vector is longer than f
        ny = len(y)
        nf = len(f)
        ymean = np.cumsum(y) / np.arange(1, ny+1)
        # Assume that the forecasts and y vector terminate at the same point
        y = y[-nf:]
        ymean = ymean[-nf:]
    return np.mean(((y.reshape(-1) - f.reshape(-1)) ** 2 / (ymean.reshape(-1) ** 2)))
