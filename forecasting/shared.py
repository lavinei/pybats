import numpy as np
import scipy as sc
from scipy.special import digamma

# I need this helper in a module file for pickle reasons ...
def transformer(ft, qt, fn1, fn2):
    return np.array([np.exp(fn1(ft, np.sqrt(qt))), np.exp(fn2(ft, np.sqrt(qt)))])
    # return np.array([np.exp(fn1(ft, np.sqrt(qt))[0,0]), np.exp(fn2(ft, np.sqrt(qt))[0,0])])

def _interp_fn_log_wrap(ft, qt, _interp_fn_log):
    return _interp_fn_log(ft, qt)[0,0]

def gamma_interp_fn_log_alpha(ft, qt, _interp_fn_log_alpha):
    return _interp_fn_log_alpha(qt)

def gamma_interp_fn_log_beta(ft, qt, _interp_fn_log_alpha):
    alpha = np.exp(_interp_fn_log_alpha(qt))
    return digamma(alpha) - ft

def trigamma(x):
    return sc.special.polygamma(x=x, n=1)

