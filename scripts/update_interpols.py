import numpy as np

import sys

sys.path.insert(0, '.')

from forecasting.conjugates import bern_conjugate_params, pois_conjugate_params, pois_alpha_param
from forecasting.shared import transformer, _interp_fn_log_wrap, gamma_interp_fn_log_beta, gamma_interp_fn_log_alpha
from scipy import interpolate
from scipy.special import digamma

import pickle
import zlib
from functools import partial


def _get_grid(ft_lb, ft_ub, qt_lb, qt_ub, num=250j):
    ft, qt = np.mgrid[ft_lb:ft_ub:num, qt_lb:qt_ub:num]
    return ft.flatten(), qt.flatten()


def interp_gamma():
    """
    We will interpolate based on reasonable values of the mean and std dev.

    For gamma, -4 is pretty small on the log scale and 8 is quite large. Similarly,
    0.01 is a very small std dev and 2 is a pretty big spread (on the log scale).

    :return: a stand in fn for interpolating the conjugate map
    """
    # get the grid we will search
    ft_lb, ft_ub, qt_lb, qt_ub = -4, 8, 0.01, 2
    ft, qt = _get_grid(ft_lb, ft_ub, qt_lb, qt_ub)

    # do a bunch of solves
    z = np.empty((len(ft), 2))
    for i in range(len(ft)):
        alpha = pois_alpha_param(qt[i]**2, alpha=1.)
        beta = np.exp(digamma(alpha) - ft[i])
        z[i, :] = np.log(np.array([alpha, beta])).reshape(-1)

    _interp_fn_log_alpha = interpolate.LSQUnivariateSpline(qt[:250], z[:250, 0],
                                                           np.linspace(qt_lb * 1.01, qt_ub * 0.99, 50),
                                                           bbox=[0., qt_ub + 3],
                                                           k=2)


    # transform to original scale and variance instead of std dev
    fn = partial(transformer,
                 fn1=partial(gamma_interp_fn_log_alpha, _interp_fn_log_alpha = _interp_fn_log_alpha),
                 fn2=partial(gamma_interp_fn_log_beta, _interp_fn_log_alpha = _interp_fn_log_alpha))

    fn.ft_lb, fn.ft_ub, fn.qt_lb, fn.qt_ub = -5, 9, 0.01 ** 2, 2 ** 2
    return fn


def interp_beta():
    """
    We will interpolate based on reasonable values of the mean and std dev.

    For beta, -4 is pretty small on the log scale and 4 is quite large. Similarly,
    0.01 is a very small std dev and 2 is a pretty big spread (on the log scale).

    :return: a stand in fn for interpolating the conjugate map
    """
    # get the grid we will search
    ft_lb, ft_ub, qt_lb, qt_ub = -6, 6, 0.05, 2
    ft, qt = _get_grid(ft_lb, ft_ub, qt_lb, qt_ub, num=350j)

    # do a bunch of solves
    z = np.empty((len(ft), 2))
    for i in range(len(ft)):
        z[i, :] = np.log(bern_conjugate_params(ft[i], qt[i] ** 2, alpha=1, beta=1, interp=False))

    npts = 100

    _interp_fn_log_alpha = interpolate.LSQBivariateSpline(ft, qt, z[:, 0],
                                                          np.linspace(ft_lb * 0.99, ft_ub * 0.99, npts),
                                                          np.linspace(qt_lb * 1.01, qt_ub * 0.99, npts),
                                                          bbox=[ft_lb - 2, ft_ub + 2, 0., qt_ub + 3],
                                                          kx=2, ky=2)

    _interp_fn_log_beta = interpolate.LSQBivariateSpline(ft, qt, z[:, 1],
                                                         np.linspace(ft_lb * 0.99, ft_ub * 0.99, npts),
                                                         np.linspace(qt_lb * 1.01, qt_ub * 0.99, npts),
                                                         bbox=[ft_lb - 2, ft_ub + 2, 0., qt_ub + 3],
                                                         kx=2, ky=2)

    # transform to original scale and variance instead of std dev
    fn = partial(transformer,
                 fn1=partial(_interp_fn_log_wrap, _interp_fn_log = _interp_fn_log_alpha),
                 fn2=partial(_interp_fn_log_wrap, _interp_fn_log = _interp_fn_log_beta))

    fn.ft_lb, fn.ft_ub, fn.qt_lb, fn.qt_ub = -7, 7, np.round(0.05 ** 2, 4), 2 ** 2
    return fn


if __name__ == '__main__':
    print('Calculating Gamma Interpolation FNs')
    interp_gamma = interp_gamma()
    print('Writing FNs to disk')
    with open('interp_gamma.pickle.gzip', 'wb') as fl:
        fl.write(zlib.compress(pickle.dumps(interp_gamma)))

    print('Calculating Beta Interpolation FNs')
    interp_beta = interp_beta()
    print('Writing FNs to disk')
    with open('interp_beta.pickle.gzip', 'wb') as fl:
        fl.write(zlib.compress(pickle.dumps(interp_beta)))
