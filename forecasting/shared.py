import numpy as np
import scipy as sc

# I need this helper in a module file for pickle reasons ...
def transformer(ft, qt, fn1, fn2):
    return np.exp(fn1(ft, np.sqrt(qt))), np.exp(fn2(ft, np.sqrt(qt)))

def trigamma(x):
    return sc.special.polygamma(x=x, n=1)

