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

def gamma_alpha_approx(x, qt):
    x = x**2
    return np.array([trigamma(x=x[0]) - qt]).reshape(-1)

def pois_alpha_param(qt, alpha=1.):
    sol = opt.root(partial(gamma_alpha_approx, qt=qt), x0=np.sqrt(np.array([alpha])), method='lm')
    return sol.x ** 2

def gamma_solver(ft, qt, alpha=1., beta=1.):

    # If q_t is is small, can use an approximation
    if qt < 0.0001:
        alpha = 1/qt
        beta = np.exp(digamma(alpha) - ft)
        return np.array([alpha, beta])

    # all else fails, do the optimization for alpha, followed by an exact soln for beta
    alpha = pois_alpha_param(qt)[0]
    beta = np.exp(digamma(alpha) - ft)
    return np.array([alpha, beta])

def beta_solver(ft, qt, alpha=1., beta=1.):

    # If qt is small, likely consistent with a large alpha, beta - can use an approximation
    # Ref: West & Harrison, pg. 530
    alpha = (1 / qt) * (1 + np.exp(ft))
    beta = (1 / qt) * (1 + np.exp(-ft))
    if qt < 0.0025:
        return np.array([alpha, beta])


    # all else fails, do the optimization
    sol = opt.root(partial(beta_approx, ft=ft, qt=qt), x0=np.sqrt(np.array([alpha, beta])), method='lm')
    return sol.x ** 2


# generic conj function
def conj_params(ft, qt, alpha=1., beta=1., interp=False, solver_fn=None, interp_fn=None):
    # the shape of these can vary a lot, so standardizing here.
    ft, qt = np.ravel(ft)[0], np.ravel(qt)[0]

    # do we want to interpolate?
    if interp and interp_fn is not None:
        # we may be asking for a value that's outside the interp range
        if interp_fn.ft_lb < ft < interp_fn.ft_ub and \
                interp_fn.qt_lb**2 < qt < interp_fn.qt_ub**2:
            return interp_fn(ft, qt)
    # all else fails, do the optimization
    return solver_fn(ft, qt, alpha, beta)


# specific conjugate params functions
bern_conjugate_params = partial(conj_params, solver_fn=beta_solver, interp_fn=interp_beta, interp=True)
pois_conjugate_params = partial(conj_params, solver_fn=gamma_solver, interp_fn=interp_gamma, interp=True)
bin_conjugate_params = partial(conj_params, solver_fn=beta_solver, interp_fn=interp_beta, interp=True)

