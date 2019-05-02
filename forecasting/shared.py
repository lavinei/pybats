import numpy as np
import scipy as sc
from scipy.special import digamma


# I need this helper in a module file for pickle reasons ...
def transformer(ft, qt, fn1, fn2):
    return np.exp(np.ravel(fn1(ft, np.sqrt(qt), grid=False))), \
           np.exp(np.ravel(fn2(ft, np.sqrt(qt), grid=False)))


def gamma_transformer(ft, qt, fn):
    a = np.ravel(np.exp(fn(np.sqrt(qt))))
    b = np.exp(digamma(a) - ft)
    return a, b


def trigamma(x):
    return sc.special.polygamma(x=x, n=1)
