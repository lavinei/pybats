import numpy as np

import sys

sys.path.insert(0, '.')

from pybats_nbdev.conjugates import bern_conjugate_params, pois_conjugate_params, pois_alpha_param
from pybats_nbdev.shared import transformer, trigamma, gamma_transformer
from scipy import interpolate
from scipy import optimize as opt
from scipy.special import digamma

import pickle
import zlib
from functools import partial


def _get_grid(ft_lb, ft_ub, qt_lb, qt_ub, num=300j):
    # get a grid of means on a linear scale and std devs on log scale
    ft, qt = np.mgrid[ft_lb:ft_ub:num, np.log(qt_lb):np.log(qt_ub):num]
    return ft.flatten(), np.exp(qt.flatten())

def interp_gamma():
    """
    We will interpolate based on reasonable values of the std dev and use the exact mean.

    For the std dev, 0.001 is a very small std dev and 4 is a pretty big spread (on the log scale).

    :return: a stand in fn for interpolating the conjugate map
    """
    # get the grid we will search
    qt_lb, qt_ub = 0.0001, 4
    qt = np.exp(np.linspace(np.log(qt_lb), np.log(qt_ub), 5000))

    # do a bunch of solves
    z = np.empty(len(qt))
    for i in range(len(qt)):
        z[i] = np.log(pois_alpha_param(qt[i] ** 2))

    knots = np.exp(np.linspace(np.log(qt[5]), np.log(qt[-5]), 100))
    bbox = [0, qt_ub + 2]
    _interp_fn_log_alpha = interpolate.LSQUnivariateSpline(qt, z,
                                                           knots, bbox=bbox, k=1)

    # transform to original scale and variance instead of std dev
    fn = partial(gamma_transformer, fn=_interp_fn_log_alpha)
    fn.ft_lb, fn.ft_ub, fn.qt_lb, fn.qt_ub = -np.inf, np.inf, bbox[0], bbox[1]
    return fn


def interp_beta():
    """
    We will interpolate based on reasonable values of the mean and std dev.

    For beta, -8 is pretty small on the log scale and 8 is quite large. Similarly,
    0.001 is a very small std dev and 4 is a pretty big spread (on the log scale).

    :return: a stand in fn for interpolating the conjugate map
    """
    # get the grid we will search
    ft_lb, ft_ub, qt_lb, qt_ub = -8, 8, 0.0001, 4
    ft, qt = _get_grid(ft_lb, ft_ub, qt_lb, qt_ub)

    # do a bunch of solves
    z = np.empty((len(ft), 2))
    for i in range(len(ft)):
        z[i, :] = np.log(bern_conjugate_params(ft[i], qt[i] ** 2, alpha=1, beta=1, interp=False))

    # ft and qt knots
    num_knots = 75
    ftv, qtv = np.sort(np.unique(ft)), np.sort(np.unique(qt))
    # make sure there are a few points outside the knots
    ft_knots = np.linspace(ftv[3], ftv[-3], num_knots)
    qt_knots = np.exp(np.linspace(np.log(qtv[3]), np.log(qtv[-3]), num_knots))
    bbox = [ft_lb - 2, ft_ub + 2, 0., qt_ub + 2]

    _interp_fn_log_alpha = interpolate.LSQBivariateSpline(ft, qt, z[:, 0],
                                                          ft_knots, qt_knots,
                                                          bbox=bbox,
                                                          kx=1, ky=1)
    _interp_fn_log_beta = interpolate.LSQBivariateSpline(ft, qt, z[:, 1],
                                                         ft_knots, qt_knots,
                                                         bbox=bbox,
                                                         kx=1, ky=1)
    # transform to original scale and variance instead of std dev
    fn = partial(transformer, fn1=_interp_fn_log_alpha, fn2=_interp_fn_log_beta)
    fn.ft_lb, fn.ft_ub, fn.qt_lb, fn.qt_ub = bbox
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
