import numpy as np

from .dglm import dlm, pois_dglm, bern_dglm, bin_dglm
import statsmodels.api as sm


def define_dglm(Y, X, family="normal", n=None,
                ntrend=1, nhol=0,
                seasPeriods=[7], seasHarmComponents = [[1, 2, 3]],
                deltrend = .995, delregn =.995, delseas = .999, delVar = 0.999, delhol=1,
                n0 = 1, s0 = 1,
                a0=None, R0=None,
                adapt_discount='info', discount_forecast=False,
                prior_length=None, return_aR=False,
                **kwargs):
    """
    A helper function to define a DGLM.

    This function is especially useful if you do not know how to specifify a prior mean and variance (a0, R0) for the state vector.

    .. code::

        mod = define_dglm(Y, X, family='poisson',       # Observation vector Y, regression predictors X, and the exponential family.
                          prior_length=21,              # Number of observations to use in defining prior
                          ntrend=1,                     # Number of 'trend' or polynomial components. 1=intercept (local level). 2=intercept and local slope.
                          nhol=0,                       # Number of holiday indicators in the model. These are regression components.
                          seasPeriods=[7],              # List of the periods of seasonal components. This includes a seasonal component with period 7, which is typical for day-of-week seasonality.
                          seasHarmComponents=[[1,2,3]]  # List of harmonic components for each seasonal component. These components go up to period/2, rounded down. So we include the 1st, 2nd, and 3rd component for a seasonality with period 7.
                          )

    This function is called automatically within 'analysis' if a model prior is not specified.

    :param Y: Observation array used to define prior
    :param X: Predictor array used to define prior (includes indicator columns for holidays)
    :param ntrend: Number of trend components. 1 = Intercept only. 2 = Intercept + slope
    :param nhol: Number of holiday components
    :param seasPeriods: List of periods for seasonal components
    :param seasHarmComponents: List of harmonic components included in each seasonal component
    :param deltrend: Discount factor on trend components
    :param delregn: Discount factor on regression components
    :param delseas: Discount factor on seasonal components
    :param delVar: Discount factor on stochastic volatility (observation error)
    :param delhol: Discount factor on holiday components
    :param n0: Prior 'sample size' for stochastic volatility
    :param s0: Prior standard deviation of stochastic volatility
    :param a0: Prior state vector mean
    :param R0: Prior state vector covariance
    :param adapt_discount: Optional. Can be 'info' or 'positive_regn'. Ways to adapt discount factors, and prevent exploding variance.
    :param discount_forecast: Optional, boolean. Should forecasts be discounted? If yes, variance added to state vector with state evolution equation.
    :param prior_length: Optional, number of rows from Y, X to use. Otherwise all are used
    :param kwargs:
    :return: Returns an initialized DGLM
    """

    if a0 is None or R0 is None:
        # Inferring the number of observations to use for the prior
        if prior_length is not None:
            n_obs = prior_length
        else:
            n_obs = len(Y)

        # Adding an intercept to the X matrix, if it doesn't already exist
        if X is None and ntrend >= 1:
            X_withintercept = np.ones([n_obs, 1])
        elif ntrend >= 1:
            if len(X.shape) == 1:
                X = X.reshape(-1,1)

            if not np.all(X[:,0] == 1):
                X_withintercept = np.c_[np.ones([n_obs, 1]), X[:n_obs]]
            else:
                X_withintercept = X[:n_obs]

        # Selecting only the correct number of observations (relevant if prior_length is given
        Y = Y[:n_obs]
        if n is not None:
            n = n[:n_obs]

        # Infer the number of regression and holiday components
        nregn = ncol(X_withintercept) - nhol - 1
        nseas = 2 * sum(map(len, seasHarmComponents))

        # Learn a prior based on the first 'prior_length' observations
        if family == "normal":
            prior_mean, prior_cov, p = define_dlm_params(Y, X_withintercept)
        elif family == "poisson":
            prior_mean, prior_cov, p = define_pois_params(Y, X_withintercept)
        elif family == "bernoulli":
            prior_mean, prior_cov, p = define_bern_params(Y, X_withintercept)
        elif family == "binomial":
            prior_mean, prior_cov, p = define_bin_params(Y, n, X_withintercept)

        # Define a standard prior - setting latent factor priors at 1
        # Unless prior mean (a0) and prior variance (R0) are supplied as arguments
        prior = [[prior_mean[0]], [0] * (ntrend - 1), [*prior_mean[1:]], [0] * nseas]
        if a0 is None:
            a0 = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
        if R0 is None:
            R0 = np.identity(a0.shape[0])
            idx = [i for i in range(p + ntrend - 1)]
            for j in range(1, ntrend):
                idx.pop(j)
            R0[np.ix_(idx, idx)] = prior_cov

        # Add variance to holiday indicators - few observations, may be significantly different than other days
        ihol = range(ntrend + nregn, ntrend + nregn + nhol)
        for idx in ihol:
            R0[idx, idx] = R0[idx, idx] * 2
    else:
        # Infer the number of regression and holiday components
        p = len(a0)
        nseas = 2 * sum(map(len, seasHarmComponents))
        nregn = p - ntrend - nhol - nseas


    if return_aR:
        return a0, R0, nregn

    if family == "normal":
        mod = dlm(a0=a0, R0=R0,
                  nregn=nregn,
                  ntrend=ntrend,
                  nhol=nhol,
                  seasPeriods=seasPeriods,
                  seasHarmComponents=seasHarmComponents,
                  deltrend=deltrend, delregn=delregn,
                  delseas=delseas, delhol=delhol,
                  n0=n0, s0=s0, delVar=delVar,
                  adapt_discount=adapt_discount,
                  discount_forecast = discount_forecast)
    elif family == "poisson":
        if kwargs.get('rho') is not None:
            rho = kwargs.get('rho')
        else:
            rho = 1
        mod = pois_dglm(a0=a0, R0=R0,
                        nregn=nregn,
                        ntrend=ntrend,
                        nhol=nhol,
                        seasPeriods=seasPeriods,
                        seasHarmComponents=seasHarmComponents,
                        deltrend=deltrend, delregn=delregn,
                        delseas=delseas, delhol=delhol,
                        adapt_discount=adapt_discount,
                        discount_forecast = discount_forecast,
                        rho = rho)
    elif family == "bernoulli":
        mod = bern_dglm(a0=a0, R0=R0,
                        nregn=nregn,
                        ntrend=ntrend,
                        nhol=nhol,
                        seasPeriods=seasPeriods,
                        seasHarmComponents=seasHarmComponents,
                        deltrend=deltrend, delregn=delregn,
                        delseas=delseas, delhol=delhol,
                        adapt_discount=adapt_discount,
                        discount_forecast = discount_forecast)
    elif family == "binomial":
        mod = bin_dglm(a0=a0, R0=R0,
                       nregn=nregn,
                       ntrend=ntrend,
                       nhol=nhol,
                       seasPeriods=seasPeriods,
                       seasHarmComponents=seasHarmComponents,
                       deltrend=deltrend, delregn=delregn,
                       delseas=delseas, delhol=delhol,
                       adapt_discount=adapt_discount,
                       discount_forecast = discount_forecast)


    return mod


def define_dlm_params(Y, X=None):
    n = len(Y)
    p = ncol(X)
    g = max(2, int(n / 2))

    linear_mod = sm.OLS(Y, X).fit()

    dlm_mean = linear_mod.params
    dlm_cov = fill_diag((g / (1 + g)) * linear_mod.cov_params())

    return dlm_mean, dlm_cov, p


def define_bern_params(Y, X=None):
    n = len(Y)
    p = ncol(X)

    nonzeros = Y.nonzero()[0]

    g = max(2, int(n/2))
    try:
        Y_bern = np.c_[np.zeros([n, 1]), np.ones([n, 1])]
        Y_bern[Y.nonzero()[0], 0] = 1
        Y_bern[Y.nonzero()[0], 1] = 0
        bern_mod = sm.GLM(endog=Y_bern, exog=X, family=sm.families.Binomial()).fit()
        bern_params = bern_mod.params
        bern_cov = fill_diag((g/(1+g))*bern_mod.cov_params())
    except:
        if len(nonzeros) > 0:
            bernmean = len(nonzeros) / (n + 1)
            bernmean = np.log(bernmean / (1 - bernmean))
            bern_params = np.zeros(p)
            bern_params[0] = bernmean
        else:
            bern_params = np.zeros(p)
        bern_cov = np.identity(p)

    return bern_params, bern_cov, p


def define_bin_params(Y, n, X=None):
    n_obs = len(Y)
    p = ncol(X)

    g = max(2, int(n_obs / 2))
    try:
        bin_mod = sm.GLM(endog=np.c_[Y, n], exog=X, family=sm.families.Binomial()).fit()
        bin_params = bin_mod.params
        bin_cov = fill_diag((g/(1+g))*bin_mod.cov_params())
    except:
        if np.sum(Y) > 0:
            binmean = np.sum(Y) / np.sum(n)
            binmean = np.log(binmean / (1 - binmean))
            bin_params = np.zeros(p)
            bin_params[0] = binmean
        else:
            bin_params = np.zeros(p)
            bin_params[0] = np.max([-3, -np.sum(n)])
        bin_cov = np.identity(p)

    return bin_params, bin_cov, p


def define_pois_params(Y, X=None):
    n = len(Y)
    p = ncol(X)

    g = max(2, int(n/2))
    try:
        pois_mod = sm.GLM(Y, X,
                          family=sm.families.Poisson()).fit()
        pois_params = pois_mod.params
        pois_cov = fill_diag((g/(1+g))*pois_mod.cov_params())
    except:
        pois_params = np.zeros(p)
        pois_cov = np.identity(p)

    return pois_params, pois_cov, p


def ncol(x):
    if x is None:
        return 0
    if len(np.shape(x)) == 1:
        return 1
    else:
        return np.shape(x)[1]


def fill_diag(cov):
    diag = cov.diagonal().copy()
    diag[diag == 0] = 1
    np.fill_diagonal(cov, diag)
    return cov