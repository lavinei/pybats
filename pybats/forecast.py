## The standard DGLM forecast functions
import numpy as np
from scipy import stats
from scipy.special import gamma

from pybats.update import update_F


def forecast_aR(mod, k):
    Gk = np.linalg.matrix_power(mod.G, k - 1)
    a = Gk @ mod.a
    R = Gk @ mod.R @ Gk.T
    if mod.discount_forecast:
        R += (k - 1) * mod.W
    return a, R


def forecast_R_cov(mod, k1, k2):
    """
    :param mod: model
    :param k1: 1st Forecast Horizon (smaller)
    :param k2: 2nd Forecast Horizon (larger)
    :return: State vector covariance across k1, k2. West & Harrison the covariance is defined as Ct(k,j), pg. 106
    """
    if k2 < k1:
        tmp = k1
        k1 = k2
        k2 = tmp
    Gk = np.linalg.matrix_power(mod.G, k2 - k1)
    a, Rk1 = forecast_aR(mod, k1)
    return Gk @ Rk1


def forecast_marginal(mod, k, X = None, nsamps = 1, mean_only = False, state_mean_var = False):
    """
    Forecast function k steps ahead (marginal)
    """
    # Plug in the correct F values
    F = update_F(mod, X, F=mod.F.copy())

    # Evolve to the prior for time t + k
    a, R = forecast_aR(mod, k)

    # Mean and variance
    ft, qt = mod.get_mean_and_var(F, a, R)

    if state_mean_var:
        return ft, qt
        
    # Choose conjugate prior, match mean and variance
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)
    
    if mean_only:
        return mod.get_mean(param1, param2)
        
    # Simulate from the forecast distribution
    return mod.simulate(param1, param2, nsamps)


def forecast_path(mod, k, X = None, nsamps = 1):
    """
    Forecast function for the k-step path
    k: steps ahead to forecast
    X: array with k rows for the future regression components
    nsamps: Number of samples to draw from forecast distribution
    """
        
    samps = np.zeros([nsamps, k])
        
    F = np.copy(mod.F)
        
    for n in range(nsamps):
        param1 = mod.param1
        param2 = mod.param2
        
        a = np.copy(mod.a)
        R = np.copy(mod.R)
                
        for i in range(k):

            # Plug in the correct F values
            if mod.nregn > 0:
                F = update_F(mod, X[i,:], F=F)
            # if mod.nregn > 0:
            #     F[mod.iregn] = X[i,:].reshape(mod.nregn,1)

            # Get mean and variance
            ft, qt = mod.get_mean_and_var(F, a, R)

            # Choose conjugate prior, match mean and variance
            param1, param2 = mod.get_conjugate_params(ft, qt, param1, param2)

            # Simulate next observation
            samps[n, i] = mod.simulate(param1, param2, nsamps = 1)

            # Update based on that observation
            param1, param2, ft_star, qt_star = mod.update_conjugate_params(samps[n, i], param1, param2)

            # Kalman filter update on the state vector (using Linear Bayes approximation)
            m = a + R @ F * (ft_star - ft)/qt
            C = R - R @ F @ F.T @ R * (1 - qt_star/qt)/qt
            
            # Get priors a, R for the next time step
            a = mod.G @ m
            R = mod.G @ C @ mod.G.T
            R = (R + R.T)/2
                
            # Discount information
            if mod.discount_forecast:
                R = R + mod.W

    return samps

def forecast_marginal_bindglm(mod, n, k, X=None, nsamps=1, mean_only=False):
    """
    Forecast function k steps ahead (marginal)
    """
    # Plug in the correct F values
    F = update_F(mod, X, F=mod.F.copy())
    # F = np.copy(mod.F)
    # if mod.nregn > 0:
    #     F[mod.iregn] = X.reshape(mod.nregn,1)

    # Evolve to the prior for time t + k
    a, R = forecast_aR(mod, k)

    # Mean and variance
    ft, qt = mod.get_mean_and_var(F, a, R)

    # Choose conjugate prior, match mean and variance
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)

    if mean_only:
        return mod.get_mean(n, param1, param2)

    # Simulate from the forecast distribution
    return mod.simulate(n, param1, param2, nsamps)


def forecast_path_dlm(mod, k, X=None, nsamps=1, approx=True):
    """
    Forecast function for the k-step path
    k: steps ahead to forecast
    X: array with k rows for the future regression components
    nsamps: Number of samples to draw from forecast distribution
    """

    if approx:

        mean = np.zeros([k])
        cov = np.zeros([k, k])

        F = np.copy(mod.F)

        Flist = [None for x in range(k)]
        Rlist = [None for x in range(k)]

        for i in range(k):

            # Evolve to the prior at time t + i + 1
            a, R = forecast_aR(mod, i + 1)

            Rlist[i] = R

            # Plug in the correct F values
            if mod.nregn > 0:
                F = update_F(mod, X[i, :], F=F)

            Flist[i] = np.copy(F)

            # Find lambda mean and var
            ft, qt = mod.get_mean_and_var(F, a, R)
            mean[i] = ft
            cov[i, i] = qt

            # Find covariances with previous lambda values
            for j in range(i):
                # Covariance matrix between the state vector at times j, i, i > j
                cov_ij = np.linalg.matrix_power(mod.G, i - j) @ Rlist[j]
                # Covariance between lambda at times j, i
                cov[j, i] = cov[i, j] = Flist[j].T @ cov_ij @ Flist[i]

        return multivariate_t(mean, cov, mod.n, nsamps)

    else:

        samps = np.zeros([nsamps, k])
        F = np.copy(mod.F)
        p = len(F)

        ## Initialize samples of the state vector and variance from the prior
        v = 1.0 / np.random.gamma(shape=mod.n / 2, scale=2 / (mod.n * mod.s[0]), size=nsamps)
        thetas = np.array(list(
            map(lambda var: np.random.multivariate_normal(mean=mod.a.reshape(-1), cov=var / mod.s * mod.R, size=1).T,
                v))).squeeze()

        for i in range(k):

            # Plug in the correct F values
            if mod.nregn > 0:
                F = update_F(mod, X[i, :], F=F)

            # mean
            ft = (thetas @ F).reshape(-1)

            # Simulate from the sampling model
            samps[:, i] = mod.simulate_from_sampling_model(ft, v, nsamps)

            # Evolve the state vector and variance for the next timestep
            if mod.discount_forecast:
                v = v * np.random.beta(mod.delVar * mod.n / 2, ((1 - mod.delVar) * mod.n) / 2, size=nsamps)
                thetas = np.array(list(
                    map(lambda theta, var: mod.G @ theta + np.random.multivariate_normal(mean=np.zeros(p),
                                                                                         cov=var / mod.s * mod.W,
                                                                                         size=1),
                        thetas, v))).squeeze()
            else:
                v = v
                thetas = (mod.G @ thetas.T).T

        return samps

def multivariate_t(mean, scale, nu, nsamps):
    '''
    mean = mean
    scale = covariance matrix * ((nu-2)/nu)
    nu = degrees of freedom
    nsamps = # of samples to produce
    '''
    p = len(mean)
    g = np.tile(np.random.gamma(nu/2.,2./nu, nsamps), (p, 1)).T
    Z = np.random.multivariate_normal(np.zeros(p), scale, nsamps)
    return mean + Z/np.sqrt(g)


def multivariate_t_density(y, mean, scale, nu):
    '''
    y = vector of observations
    mean = mean
    scale = covariance matrix * ((nu-2)/nu)
    nu = degrees of freedom
    '''
    y = y.reshape(-1, 1)
    mean = mean.reshape(-1, 1)
    dim = len(y)
    if dim > 1:
        constant = gamma((nu + dim) / 2) / (gamma(nu / 2) * np.sqrt((np.pi * nu) ** dim * np.linalg.det(scale)))
        dens = (1. + ((y - mean).T @ np.linalg.inv(scale) @ (y - mean)) / nu) ** (-(nu + dim) / 2)
    else:
        constant = gamma((nu + dim) / 2) / (gamma(nu / 2) * np.sqrt((np.pi * nu) ** dim * scale))
        dens = (1. + ((y - mean))**2 / (nu * scale)) ** (-(nu + dim) / 2)

    return 1. * constant * dens


def forecast_state_mean_and_var(mod, k = 1, X = None):
    """
       Forecast function that returns the mean and variance of lambda = state vector * predictors
       """
    # Plug in the correct F values
    F = update_F(mod, X, F=mod.F.copy())
    # F = np.copy(mod.F)
    # if mod.nregn > 0:
    #     F[mod.iregn] = X.reshape(mod.nregn, 1)

    # Evolve to the prior for time t + k
    a, R = forecast_aR(mod, k)

    # Mean and variance
    ft, qt = mod.get_mean_and_var(F, a, R)

    return ft, qt


def forecast_marginal_density_MC(mod, k, X = None, nsamps = 1, y = None):
    """
    Returns the log forecast density of an observation y
    """
    # Plug in the correct F values
    F = update_F(mod, X, F=mod.F.copy())
    # F = np.copy(mod.F)
    # if mod.nregn > 0:
    #     F[mod.iregn] = X.reshape(mod.nregn, 1)

    # Evolve to the prior for time t + k
    a, R = forecast_aR(mod, k)

    # Mean and variance
    ft, qt = mod.get_mean_and_var(F, a, R)

    # Choose conjugate prior, match mean and variance
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)

    # Simulate from the conjugate prior
    prior_samps = mod.simulate_from_prior(param1, param2, nsamps)

    # Get the densities
    densities = mod.sampling_density(y, prior_samps)

    # Take a Monte Carlo average, and return the mean density, on the log scale
    return np.log(np.mean(densities))

