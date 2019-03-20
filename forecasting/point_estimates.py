import numpy as np

# Note: Assume that each row is an independent forecast sample
# Columns can then be forecast horizons 1:k, or they can be different items, whichever
def mean(samps):
    return np.mean(samps, axis=0)

# Optimal for MAD or absolute deviation
def median(samps):
    return np.median(samps, axis=0)

# Optimal for APE. Always less than the median.
def m_one_median(samps):
    rows, cols = samps.shape

    forecast = np.zeros(cols)
    for c in range(cols):
        if np.all(samps[:,c] != 0):
            weights = 1/samps[:,c]
            norm = np.sum(weights)
            weights = weights / norm
            order = np.argsort(samps[:,c])
            ord_samps = samps[order,c]
            ord_weights = weights[order]
            lower = ord_samps[np.max(np.where(np.cumsum(ord_weights) < 0.5))]
            upper = ord_samps[np.min(np.where(np.cumsum(ord_weights) > 0.5))]
            forecast[c] = (upper + lower)/2
        else:
            forecast[c] = np.nan

    return forecast

# Here we get the joint one_median, where the rows are forecast samples
# And the forecast is joint across columns (usually different items)
# Optimal for WAPE. Should fall between APE and the Median.
def joint_m_one_median(samps):
    rows, cols = samps.shape
    # Remove rows that are all zero
    rowsums = np.sum(samps, axis=1)
    psamps = samps[rowsums.nonzero()[0], :]
    rowsums = rowsums[rowsums.nonzero()[0]]

    # Weight each joint sample (i.e. row) by the inverse of its sum
    weights = 1/rowsums
    norm = np.sum(weights)
    weights = weights/norm


    # Get the -1 median for each column using these joint weights
    forecast = np.zeros(cols)
    for c in range(cols):
        order = np.argsort(psamps[:,c])
        ord_samps = psamps[order, c]
        ord_weights = weights[order]
        lower = ord_samps[np.max(np.where(np.cumsum(ord_weights) < 0.5))]
        upper = ord_samps[np.min(np.where(np.cumsum(ord_weights) > 0.5))]
        forecast[c] = (upper + lower) / 2

    return forecast





