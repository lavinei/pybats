import numpy as np

# NOTE: This module assumes that samples come as 3-dimensional arrays
# This is typical output from analysis functions
# The first dimension should have independent forecast samples


def mean(samps):
    """
    Return the mean point forecasts, given samples from the analysis function.

    This forecast is theoretically optimal for minimizing mean squared error loss.

    :param samps: Forecast samples, returned from the analysis function. Will have 3-dimensions (nsamps * time * forecast horizon)
    :return: Array of mean forecasts. Will have dimension (time * forecast horizon)
    """
    return np.mean(samps, axis=0)


# Optimal for MAD or absolute deviation
def median(samps):
    """
    Return the median point forecasts, given samples from the analysis function.

    This forecast is theoretically optimal for minimizing mean absolute deviation loss.

    :param samps: Forecast samples, returned from the analysis function. Will have 3-dimensions (nsamps * time * forecast horizon)
    :return: Array of median forecasts. Will have dimension (time * forecast horizon)
    """
    return np.median(samps, axis=0)


# Utility function
def weighted_quantile(samp, weights, quantile=0.5):
    order = np.argsort(samp)
    ord_samp = samp[order]
    ord_weights = weights[order]
    lower = ord_samp[np.max(np.where(np.cumsum(ord_weights) < quantile))]
    upper = ord_samp[np.min(np.where(np.cumsum(ord_weights) > quantile))]
    return np.round((upper + lower) / 2)


# Optimal for APE. Always less than the median. Returns nan if some samples are 0.
def m_one_median(samps):
    """
    Return the (-1)-median point forecasts, given samples from the analysis function.

    This forecast is theoretically optimal for minimizing absolute percentage error loss.

    :param samps: Forecast samples, returned from the analysis function. Will have 3-dimensions (nsamps * time * forecast horizon)
    :return: Array of (-1)-median forecasts. Will have dimension (time * forecast horizon)
    """
    def m_one_median(samp):
        nz = samp.nonzero()[0]
        weights = 1/samp[nz]
        norm = np.sum(weights)
        weights = weights/norm
        return weighted_quantile(samp[nz], weights)

    forecast = np.apply_along_axis(m_one_median, 0, samps)

    return forecast


# Here we get the joint one_median, where the rows are forecast samples
# Assume that the forecast is 'joint' across the last dimension
def joint_m_one_median(samps):

    def joint_m_one_median(samp):
        rows, cols = samp.shape
        # Remove rows that are all zero
        rowsums = np.sum(samp, axis=1)
        psamp = samp[rowsums.nonzero()[0], :]
        rowsums = rowsums[rowsums.nonzero()[0]]

        # Weight each joint sample (i.e. row) by the inverse of its sum
        weights = 1 / rowsums
        norm = np.sum(weights)
        weights = weights / norm

        # Get the -1 median for each column using these joint weights
        forecast = np.zeros(cols)
        for c in range(cols):
            forecast[c] = weighted_quantile(psamp[:, c], weights)

        return forecast

    if samps.ndim == 2:
        return joint_m_one_median(samps)
    elif samps.ndim == 3:
        return np.array(list(map(joint_m_one_median, samps.transpose([1,0,2]))))


# For the constrained point forecasts
# F is a vector of constraints for the totals across the 3rd dimension of 'samps'
# Expected dimensions are: nsamps x time x (forecast horizon or items)
def constrained_mean(samps, F):
    means = np.mean(samps, axis=0)
    n = means.shape[1]
    diff = (F - np.sum(means, axis=1))/n
    return means + diff.reshape(-1,1)


def constrained_median(samps, F):
    if samps.ndim == 2:
        samps = np.expand_dims(samps, axis=1)

    # Initialize values
    forecast = median(samps)
    times = forecast.shape[0]
    lambd = np.zeros(times)

    # Iterate until a solution is found for each lambda
    tol = 1
    eps = 1E-2
    max_shift = 5E-2
    iter = 0
    max_iter = 50
    diff = F - np.sum(forecast, axis=1)
    test = np.abs(diff) > tol

    while np.any(test):
        shift = np.abs(eps*diff)
        shift[shift > max_shift] = max_shift
        lambd = lambd + np.sign(diff)*shift
        percentiles = 100*(1+lambd)/2
        for idx, p in enumerate(percentiles):
            if test[idx]:
                forecast[idx,:] = np.percentile(samps[:,idx,:], p, axis=0, interpolation='nearest')
        diff = F - np.sum(forecast, axis=1)
        test = np.abs(diff) > tol
        iter += 1
        if iter > max_iter:
           break
    return forecast


def constrained_joint_m_one_median(samps, F):


    def constrained_joint_m_one_median(samp, F):
        #if samp.ndim == 2:
        #    samp = np.expand_dims(samp, axis=1)

        # Remove joint samples that are all 0
        rowsums = np.sum(samp, axis=1)
        nz = rowsums.nonzero()[0]
        samp = samp[nz,:]
        rowsums = rowsums[nz]
        # Find weights
        weights = 1 / rowsums
        norm = np.sum(weights)
        weights = weights / norm

        # Initialize value
        forecast = joint_m_one_median(samp).reshape(1,-1)
        times = forecast.shape[0]
        lambd = np.zeros(times)

        # Iterate until a solution is found for each lambda
        tol = 1
        eps = 1E-2
        max_shift = 5E-2
        iter = 0
        max_iter = 50
        diff = F - np.sum(forecast)
        test = np.abs(diff) > tol

        while np.any(test):
            shift = np.abs(eps * diff)
            if shift > max_shift:
                shift = max_shift
            lambd = lambd + np.sign(diff) * shift
            percentile = 100 * (1 + lambd) / 2
            forecast = np.array(list(map(lambda s: weighted_quantile(s, weights, percentile/100),
                                                 samp.T)))
            diff = F - np.sum(forecast)
            test = np.abs(diff) > tol
            iter += 1
            if iter > max_iter:
                break
        return forecast.reshape(1,-1)

    if samps.ndim == 2:
        samps = np.expand_dims(samps, axis=1)

    return np.array(list(map(lambda samp, F: constrained_joint_m_one_median(samp, F),
                             samps.transpose([1, 0, 2]),
                             F)))[:,0,:]

