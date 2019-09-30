import numpy as np

from .dglm import dlm, dglm, pois_dglm, bern_dglm
import statsmodels.api as sm


def define_dglm(Y, X, family="normal",
                      ntrend=1, nlf=0, nhol=0,
                      seasPeriods=[7], seasHarmComponents = [[1, 2, 3]],
                      deltrend = .995, delregn =.995, delseas = .999, dellf=.999, delVar = 0.999, delhol=1,
                      n0 = 1, s0 = 1, a0=None, R0=None,
                      adapt_discount=False, prior_length=None,
                      **kwargs):
    """
    :param Y: Observation array used to define prior
    :param X: Predictor array used to define prior
    :param ntrend: Number of trend components. 1 = Intercept only. 2 = Intercept + slope
    :param nlf: Number of latent factor components
    :param nhol: Number of holiday components
    :param seasPeriods: List of periods for seasonal components
    :param seasHarmComponents: List of harmonic components included in each seasonal component
    :param deltrend: Discount factor on trend components
    :param delregn: Discount factor on regression components
    :param delseas: Discount factor on seasonal components
    :param dellf: Discount factor on latent factor components
    :param delVar: Discount factor on stochastic volatility (observation error)
    :param delhol: Discount factor on holiday components
    :param n0: Prior 'sample size' for stochastic volatility
    :param s0: Prior standard deviation of stochastic volatility
    :param a0: Prior state vector mean
    :param R0: Prior state vector covariance
    :param adapt_discount: Optional. Can be 'info' or 'positive_regn'. Ways to adapt discount factors, and prevent exploding variance.
    :param prior_length: Optional, number of rows from Y, X to use. Otherwise all are used
    :param kwargs:
    :return: Returns an initialized DGLM
    """

    if prior_length is not None:
        if prior_length > 0:
            Y = Y[:prior_length]
            X = X[:prior_length]

    # Infer the number of regression and holiday components
    nregn = ncol(X) - nhol
    nseas = 2 * sum(map(len, seasHarmComponents))

    # Learn a prior based on the first 'prior_length' observations
    if family == "normal":
        prior_mean, prior_cov, p = define_dlm_params(Y, X)
    elif family == "poisson":
        prior_mean, prior_cov, p = define_pois_params(Y, X)
    elif family == "bernoulli":
        prior_mean, prior_cov, p = define_bern_params(Y, X)

    # Define a standard prior - setting latent factor priors at 1
    # Unless prior mean (a0) and prior variance (R0) are supplied as arguments
    prior = [[prior_mean[0]], [0] * (ntrend - 1), [*prior_mean[1:]], [0] * nseas, [1] * nlf]
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

    if family == "normal":
        mod = dlm(a0=a0, R0=R0,
              nregn=nregn,
              ntrend=ntrend,
              nlf=nlf,
              nhol=nhol,
              seasPeriods=seasPeriods,
              seasHarmComponents=seasHarmComponents,
              deltrend=deltrend, delregn=delregn,
              delseas=delseas, delhol=delhol,
              dellf=dellf,
              n0=n0, s0=s0, delVar=delVar,
              adapt_discount=adapt_discount)
    elif family == "poisson":
        mod = pois_dglm(a0=a0, R0=R0,
                  nregn=nregn,
                  ntrend=ntrend,
                  nlf=nlf,
                  nhol=nhol,
                  seasPeriods=seasPeriods,
                  seasHarmComponents=seasHarmComponents,
                  deltrend=deltrend, delregn=delregn,
                  delseas=delseas, delhol=delhol,
                  dellf=dellf,
                  adapt_discount=adapt_discount)
    elif family == "bernoulli":
        mod = bern_dglm(a0=a0, R0=R0,
                        nregn=nregn,
                        ntrend=ntrend,
                        nlf=nlf,
                        nhol=nhol,
                        seasPeriods=seasPeriods,
                        seasHarmComponents=seasHarmComponents,
                        deltrend=deltrend, delregn=delregn,
                        delseas=delseas, delhol=delhol,
                        dellf=dellf,
                        adapt_discount=adapt_discount)



    return mod





def define_dlm_params(Y, X=None):
    n = len(Y)
    p = 1 + ncol(X)
    g = max(2, int(n / 2))

    if X is None:
        X = np.ones([n, 1])
    else:
        X = sm.add_constant(X)

    linear_mod = sm.OLS(Y, X).fit()


    dlm_mean = linear_mod.params
    dlm_cov = fill_diag((g / (1 + g)) * linear_mod.cov_params())

    return dlm_mean, dlm_cov, p


def define_bern_params(Y, X=None):
    n = len(Y)
    p = 1 + ncol(X)

    if X is None:
        X_bern = np.ones([n, 1])
    else:
        X_bern = np.c_[np.ones([n, 1]), X]
    nonzeros = Y.nonzero()[0]

    g = max(2, int(n/2))
    try:
        Y_bern = np.c_[np.zeros([n, 1]), np.ones([n, 1])]
        Y_bern[Y.nonzero()[0], 0] = 1
        Y_bern[Y.nonzero()[0], 1] = 0
        bern_mod = sm.GLM(endog=Y_bern, exog=X_bern, family=sm.families.Binomial()).fit()
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


def define_pois_params(Y, X=None):
    n = len(Y)
    p = 1 + ncol(X)

    if X is None:
        X = np.ones([n, 1])
    else:
        X = np.c_[np.ones([n, 1]), X]

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


def define_model_from_list(mod_list):
    """
    :param mod_list: List of models for similar outcome. All models in the list must have identical components & state vector size.
    :return: An initialized model, averaged across the model list
    """
    mod = None

    # Assume that all models in the list have the same number of trend, regn, lf, holiday, seasonal components
    if isinstance(mod_list[0], dlm):
        # Discount factors averaged across models
        components = ['ntrend','nregn', 'nlf', 'nhol', 'seasPeriods','seasHarmComponents']
        kwargs = {name:mod_list[0].__dict__.get(name) for name in components}
        kwargs.update({'nregn':kwargs.get('nregn') - kwargs.get('nhol')})

        discount_factors = ['deltrend', 'delregn', 'delseas', 'dellf', 'delVar', 'delhol']
        kwargs.update({name:get_mean(mod_list, name) for name in discount_factors})

        # Precision weighted average to get state vector mean
        kwargs.update({'a0' : get_mean_pw(mod_list, 'a', 'R')})

        # Define covariance matrix as the average across model list, inflated
        kwargs.update({'R0' : get_mean(mod_list, 'R')*5})

        # Define model
        kwargs.update({'s0':get_mean(mod_list, 's'),
                       'n0':get_mean(mod_list, 's')/2,
                       'adapt_discount':mod_list[0].adapt_discount})
        mod = dlm(**kwargs)

    if isinstance(mod_list[0], dglm) and not isinstance(mod_list[0], dlm):
        # Discount factors averaged across models
        components = ['ntrend', 'nregn', 'nlf', 'nhol', 'seasPeriods', 'seasHarmComponents']
        kwargs = {name: mod_list[0].__dict__.get(name) for name in components}

        discount_factors = ['deltrend', 'delregn', 'delseas', 'dellf', 'delhol']
        kwargs.update({name: get_mean(mod_list, name) for name in discount_factors})

        # Precision weighted average to get state vector mean
        kwargs.update({'a0': get_mean_pw(mod_list, 'a', 'R')})

        # Define covariance matrix as the average across model list, inflated
        kwargs.update({'R0': get_mean(mod_list, 'R') * 5})

        # Define model
        kwargs.update({'adapt_discount': mod_list[0].adapt_discount})
        mod = dlm(**kwargs)

    return mod


def get_mean(mod_list, name):
    if isinstance(mod_list[0].__dict__.get(name), np.ndarray):
        m = np.array([mod.__dict__.get(name) for mod in mod_list])
        return np.mean(m, axis=0)
    else:
        return np.mean([mod.__dict__.get(name) for mod in mod_list])


def get_mean_pw(mod_list, mean_name, var_name):
    if len(mod_list[0].__dict__.get(mean_name)) == 1:
        m = np.array([float(mod.__dict__.get(mean_name)) for mod in mod_list])
        v = np.array([float(mod.__dict__.get(var_name)) for mod in mod_list])
        p = 1 / v
        return np.sum(m * p) / np.sum(p)
    else:
        ms = [mod.__dict__.get(mean_name) for mod in mod_list]
        vs = [mod.__dict__.get(var_name) for mod in mod_list]
        ps = [np.linalg.inv(v) for v in vs]
        m = np.sum([p @ m.reshape(-1, 1) for m, p in zip(ms, ps)], axis=0)
        v = np.linalg.inv(np.sum(ps, axis=0))
        mean = v @ m
        return mean.reshape(-1)