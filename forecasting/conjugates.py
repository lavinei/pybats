import numpy as np

from scipy.special import digamma
from scipy import optimize as opt
from functools import partial

from .shared import trigamma

import pickle
import zlib
import os

pkg_data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data'

try:
    with open(pkg_data_dir + '/interp_beta.pickle.gzip', 'rb') as fl:
        interp_beta = pickle.loads(zlib.decompress(fl.read()))

    with open(pkg_data_dir + '/interp_gamma.pickle.gzip', 'rb') as fl:
        interp_gamma = pickle.loads(zlib.decompress(fl.read()))
except:
    print('WARNING: Unable to load interpolator. Code will run slower.')
    interp_beta, interp_gamma = None, None


def beta_approx(x, ft, qt):
    x = x ** 2
    return np.array([digamma(x[0]) - digamma(x[1]) - ft,
                     trigamma(x=x[0]) + trigamma(x=x[1]) - qt]).reshape(-1)


def gamma_approx(x, ft, qt):
    x = x ** 2
    return np.array([digamma(x[0]) - np.log(x[1]) - ft, trigamma(x=x[0]) - qt]).reshape(-1)


# generic conj function
def conj_params(ft, qt, alpha=1., beta=1., interp=False, approx_fn=None, interp_fn=None):
    # do we want to interpolate?
    if interp and interp_fn is not None:
        # we may be asking for a value that's outside the interp range
        if interp_fn.ft_lb < ft < interp_fn.ft_ub and \
                interp_fn.qt_lb < qt < interp_fn.qt_ub:
            return interp_fn(ft, qt)

    # all else fails, do the optimization
    sol = opt.root(partial(approx_fn, ft=ft, qt=qt), x0=np.sqrt(np.array([alpha, beta])))
    return sol.x ** 2


# specific conjugate params functions
bern_conjugate_params = partial(conj_params, approx_fn=beta_approx, interp_fn=interp_beta)
pois_conjugate_params = partial(conj_params, approx_fn=gamma_approx, interp_fn=interp_gamma)
bin_conjugate_params = partial(conj_params, approx_fn=beta_approx, interp_fn=interp_beta)

