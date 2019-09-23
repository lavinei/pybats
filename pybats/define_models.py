import numpy as np
import pandas as pd
from .dglm import dlm, dglm
from .dcmm import dcmm
from .dbcm import dbcm
from .shared import define_holiday_regressors
import statsmodels.api as sm
from pandas.tseries.holiday import AbstractHolidayCalendar

def define_normal_dlm(Y, X,
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
    :return: Returns an initialized DLM
    """

    if prior_length is not None:
        if prior_length > 0:
            Y = Y[:prior_length]
            X = X[:prior_length]

    # Define normal DLM for total sales
    nregn = ncol(X) - nhol
    nseas = 2 * sum(map(len, seasHarmComponents))
    params_mean, params_cov, p = define_dlm_params(Y, X)

    prior = [[params_mean[0]], [0] * (ntrend-1), [*params_mean[1:]], [0] * nseas, [1] * nlf]
    if a0 is None:
        a0 = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    if R0 is None:
        R0 = np.identity(a0.shape[0])*2
        idx = [i for i in range(p + ntrend - 1)]
        for j in range(1, ntrend):
            idx.pop(j)
        R0[np.ix_(idx, idx)] = params_cov


    mod = dlm(a0 = a0, R0 = R0,
              nregn = nregn,
              ntrend = ntrend,
              nlf=nlf,
              nhol=nhol,
              seasPeriods = seasPeriods,
              seasHarmComponents = seasHarmComponents,
              deltrend = deltrend, delregn = delregn,
              delseas = delseas, delhol=delhol,
              dellf=dellf,
              n0 = n0, s0 = s0, delVar = delVar,
              adapt_discount=adapt_discount)
    
    return mod

def define_dcmm(Y, X,
                ntrend=1, nlf=0, nhol = 0, rho=1,
                seasPeriods = [7], seasHarmComponents = [[1,2,3]],
                deltrend_bern=.995, delregn_bern=.995, delseas_bern=.995, dellf_bern=.999, delhol_bern=1,
                deltrend_pois=.998, delregn_pois=.995, delseas_pois=.995, dellf_pois=.999, delhol_pois=1,
                a0_bern = None, R0_bern = None, a0_pois = None, R0_pois = None,
                interpolate=True, adapt_discount=False, prior_length = None,
                **kwargs):
    """
    :param Y: Observation array used to define prior
    :param X: Predictor array used to define prior
    :param ntrend: Number of trend components. 1 = Intercept only. 2 = Intercept + slope
    :param nlf: Number of latent factor components
    :param nhol: Number of holiday components
    :param rho: Discount factor for random effects extension in poisson DGLM (smaller rho increases variance)
    :param seasPeriods: List of periods for seasonal components
    :param seasHarmComponents: List of harmonic components included in each seasonal component
    :param deltrend_bern: Discount factor on trend components of Bernoulli DGLM
    :param delregn_bern: Discount factor on regression components of Bernoulli DGLM
    :param delseas_bern: Discount factor on seasonal components of Bernoulli DGLM
    :param dellf_bern: Discount factor on latent factor components of Bernoulli DGLM
    :param delhol_bern: Discount factor on holiday components of Bernoulli DGLM
    :param deltrend_pois: Discount factor on trend components of Poisson DGLM
    :param delregn_pois: Discount factor on regression components of Poisson DGLM
    :param delseas_pois: Discount factor on seasonal components of Poisson DGLM
    :param dellf_pois: Discount factor on latent factor components of Poisson DGLM
    :param delhol_pois: Discount factor on holiday components of Poisson DGLM
    :param a0_bern: Prior state vector mean of Bernoulli DGLM
    :param R0_bern: Prior state vector covariance of Bernoulli DGLM
    :param a0_pois: Prior state vector mean of Poisson DGLM
    :param R0_pois: Prior state vector covariance of Poisson DGLM
    :param interpolate: Bool. Interpolate in Variational Bayes step for DGLM inference, for computational speedup.
    :param adapt_discount: Optional. Can be 'info' or 'positive_regn'. Ways to adapt discount factors, and prevent exploding variance.
    :param prior_length: Optional, number of rows from Y, X to use. Otherwise all are used
    :param kwargs:
    :return: An initialized DCMM
    """

    if prior_length is not None:
        if prior_length > 0:
            Y = Y[:prior_length]
            X = X[:prior_length]

    nregn = ncol(X) - nhol
    nseas = 2*sum(map(len, seasHarmComponents))

    bern_params, bern_cov, pois_params, pois_cov, p = define_dcmm_params(Y, X)

    prior = [[bern_params[0]], [0] * (ntrend-1), [*bern_params[1:]], [0] * nseas, [1] * nlf]
    if a0_bern is None: a0_bern = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    if R0_bern is None:
        R0_bern = np.identity(a0_bern.shape[0])
        idx = [i for i in range(p + ntrend - 1)]
        for j in range(1, ntrend):
            idx.pop(j)
        R0_bern[np.ix_(idx, idx)] = bern_cov

    prior = [[pois_params[0]], [0] * (ntrend-1), [*pois_params[1:]], [0] * nseas, [1] * nlf]
    if a0_pois is None: a0_pois = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    if R0_pois is None:
        R0_pois = np.identity(a0_pois.shape[0])
        idx = [i for i in range(p + ntrend - 1)]
        for j in range(1, ntrend):
            idx.pop(j)
        R0_pois[np.ix_(idx, idx)] = pois_cov

    # Add variance to holiday indicators
    ihol = range(ntrend + nregn, ntrend + nregn + nhol)
    for idx in ihol:
        R0_bern[idx, idx] = R0_bern[idx, idx] * 4
        R0_pois[idx, idx] = R0_pois[idx, idx] * 4

    mod = dcmm(a0_bern = a0_bern, R0_bern = R0_bern,
               nregn_bern = nregn,
               ntrend_bern = ntrend,
               nlf_bern= nlf,
               nhol_bern=nhol,
               seasPeriods_bern = seasPeriods,
               seasHarmComponents_bern = seasHarmComponents,
               deltrend_bern = deltrend_bern, delregn_bern = delregn_bern,
               delseas_bern = delseas_bern,
               dellf_bern=dellf_bern,
               delhol_bern = delhol_bern,
               a0_pois = a0_pois, R0_pois = R0_pois,
               nregn_pois = nregn,
               ntrend_pois = ntrend,
               nlf_pois=nlf,
               nhol_pois=nhol,
               seasPeriods_pois = seasPeriods,
               seasHarmComponents_pois = seasHarmComponents,
               deltrend_pois = deltrend_pois, delregn_pois = delregn_pois,
               delseas_pois = delseas_pois,
               dellf_pois=dellf_pois,
               delhol_pois = delhol_pois,
               rho = rho,
               interpolate=interpolate,
               adapt_discount=adapt_discount
               )
        
    return mod

def define_dbcm(Y_transaction, X_transaction=None, Y_cascade=None, X_cascade=None, excess_baskets=[], excess_values=[],
                ntrend=1, nlf=0, nhol=0, rho=1,
                seasPeriods=[7], seasHarmComponents=[[1, 2, 3]],
                deltrend_bern=.995, delregn_bern=.995, delseas_bern=.995, dellf_bern=.999, delhol_bern=1,
                deltrend_pois=.998, delregn_pois=.995, delseas_pois=.995, dellf_pois=.999, delhol_pois=1,
                deltrend_cascade=.999, delregn_cascade=1., delseas_cascade=.999, dellf_cascade=.999, delhol_cascade=1.,
                a0_bern=None, R0_bern=None, a0_pois=None, R0_pois=None, a0_cascade=None, R0_cascade=None,
                interpolate=True, adapt_discount=False, prior_length=None,
                **kwargs):
    """
    :param Y_transaction: Observation array of transactions used to define prior
    :param X_transaction: Predictor array for transactions used to define prior
    :param Y_cascade: Observation matrix of basket sizes used to define prior
    :param X_cascade: Predictor array for basket sizes used to define prior
    :param excess_baskets: Either excess_baskets or excess_values must be defined. Excess baskets is an array of basket sizes
    in exactly the same format as the Y_cascade values. Column j indicate number of transactions with more than j items.
    :param excess_values: List of excess values. Each element gives excess basket sizes observed on that day.
    Most days should be an empty list. Some days may have multiple transactions with excess basket sizes.
    :param ntrend: Number of trend components in the DCMM for transactions. 1 = Intercept only. 2 = Intercept + slope
    :param nlf: Number of latent factor components in the DCMM for transactions
    :param nhol: Number of holiday components in the DCMM for transactions
    :param rho: Discount factor for random effects extension in the DCMM for transactions (smaller rho increases variance)
    :param seasPeriods: List of periods for seasonal components in the DCMM for transactions
    :param seasHarmComponents: List of harmonic components included in each seasonal component in the DCMM for transactions
    :param deltrend_bern: Discount factor on trend components of Bernoulli DGLM
    :param delregn_bern: Discount factor on regression components of Bernoulli DGLM
    :param delseas_bern: Discount factor on seasonal components of Bernoulli DGLM
    :param dellf_bern: Discount factor on latent factor components of Bernoulli DGLM
    :param delhol_bern: Discount factor on holiday components of Bernoulli DGLM
    :param deltrend_pois: Discount factor on trend components of Poisson DGLM
    :param delregn_pois: Discount factor on regression components of Poisson DGLM
    :param delseas_pois: Discount factor on seasonal components of Poisson DGLM
    :param dellf_pois: Discount factor on latent factor components of Poisson DGLM
    :param delhol_pois: Discount factor on holiday components of Poisson DGLM
    :param deltrend_cascade: Discount factor on trend components of cascade binomial DGLMs
    :param delregn_cascade: Discount factor on regression components of cascade binomial DGLMs
    :param delseas_cascade: Discount factor on seasonal components of cascade binomial DGLMs
    :param dellf_cascade: Discount factor on latent factor components of cascade binomial DGLMs (DEPRECATED)
    :param delhol_cascade: Discount factor on holiday components of cascade binomial DGLMs
    :param a0_bern: Prior state vector mean of Bernoulli DGLM
    :param R0_bern: Prior state vector covariance of Bernoulli DGLM
    :param a0_pois: Prior state vector mean of Poisson DGLM
    :param R0_pois: Prior state vector covariance of Poisson DGLM
    :param interpolate: Bool. Interpolate in Variational Bayes step for DGLM inference, for computational speedup.
    :param adapt_discount: Optional. Can be 'info' or 'positive_regn'. Ways to adapt discount factors, and prevent exploding variance.
    :param prior_length: Optional, number of rows from Y, X to use. Otherwise all are used
    :param kwargs:
    :return: An initialized DBCM
    """

    # Define the dcmm
    mod_dcmm = define_dcmm(Y = Y_transaction, X = X_transaction,
                           ntrend=ntrend, nlf=nlf, nhol=nhol, rho=rho,
                           seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                           deltrend_bern=deltrend_bern, delregn_bern=delregn_bern, delseas_bern=delseas_bern,
                           dellf_bern=dellf_bern, delhol_bern = delhol_bern,
                           deltrend_pois=deltrend_pois, delregn_pois=delregn_pois, delseas_pois=delseas_pois,
                           dellf_pois=dellf_pois, delhol_pois=delhol_pois,
                           a0_bern=a0_bern, R0_bern=R0_bern, a0_pois=a0_pois, R0_pois=R0_pois,
                           interpolate=interpolate, adapt_discount=adapt_discount, prior_length=prior_length)

    # Calculate the prior means for the Cascade
    def cascade_prior_mean(alpha, beta):
        alpha += 1
        beta += 1
        mean = alpha / (alpha + beta)
        logit_mean = np.log(mean / (1 - mean))
        return logit_mean

    # Calculate the prior means for the cascades
    if prior_length is not None:
        if prior_length > 0:

            ncascade = Y_cascade.shape[1]
            nregn_cascade = ncol(X_cascade)
            ntrend_cascade = 1
            pcascade = nregn_cascade + ntrend_cascade

            Yc = np.c_[Y_transaction, Y_cascade]
            Yc = np.sum(Yc[:prior_length], axis=0)
            means = [cascade_prior_mean(Yc[i + 1], Yc[i] - Yc[i + 1]) for i in range(ncascade)]
            a0_cascade = [np.zeros(pcascade).reshape(-1, 1) for i in range(ncascade)]
            for i, m in enumerate(means):
                a0_cascade[i][0] = m
            R0_cascade = [0.1 * np.identity(pcascade) for i in range(ncascade)]

            # Initialize empirically observed excess baskets
            excess = []
            if len(excess_values) == 0 and len(excess_baskets) > 0:
                counts = np.sum(excess_baskets[:prior_length, :], axis=0)
                counts[:len(counts) - 1] = counts[:len(counts) - 1] - counts[1:]
                for val, count in enumerate(counts):
                    excess.extend([val + ncascade + 1 for c in range(count)])
            else:
                for e in excess_values[:prior_length]:
                    excess.extend(e)
    else:
        if a0_cascade is None:
            if kwargs.get('pcascade') is None:
                nregn_cascade = ncol(X_cascade)
                ntrend_cascade = 1
                pcascade = nregn_cascade + ntrend_cascade
            else:
                pcascade = kwargs.get('pcascade')
            if kwargs.get('ncascade') is None:
                ncascade = Y_cascade.shape[1]
            else:
                ncascade = kwargs.get('ncascade')

            a0_cascade = [np.zeros(pcascade).reshape(-1, 1) for i in range(ncascade)]
        else:
            nregn_cascade = len(a0_cascade) - 1
            ntrend_cascade = 1

        if R0_cascade is None:
            if kwargs.get('pcascade') is None:
                nregn_cascade = ncol(X_cascade)
                ntrend_cascade = 1
                pcascade = nregn_cascade + ntrend_cascade
            else:
                pcascade = kwargs.get('pcascade')
            if kwargs.get('ncascade') is None:
                ncascade = Y_cascade.shape[1]
            else:
                ncascade = kwargs.get('ncascade')

            R0_cascade = [0.1 * np.identity(pcascade) for i in range(ncascade)]

        excess = []


    # Define the model
    mod = dbcm(mod_dcmm=mod_dcmm,
               ncascade=ncascade,
               a0_cascade=a0_cascade,  # List of length ncascade
               R0_cascade=R0_cascade,  # List of length ncascade
               nregn_cascade=nregn_cascade,
               ntrend_cascade=1,
               nlf_cascade=0,
               seasPeriods_cascade=[],
               seasHarmComponents_cascade=[],
               deltrend_cascade=deltrend_cascade, delregn_cascade=delregn_cascade,
               delseas_cascade=delseas_cascade, dellf_cascade=dellf_cascade, delhol_cascade=delhol_cascade,
               excess=excess)

    return mod

def define_dlm_params(Y, X=None):
    n = len(Y)
    p = 1+ ncol(X)
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

    return bern_params, bern_cov


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

    return pois_params, pois_cov


def define_dcmm_params(Y, X):
    """
    Helper function to initialize parameters for the Bernoulli and Poisson DGLMs within a DCMM
    :param Y:
    :param X:
    :param prior_length:
    :return:
    """
    p = 1 + ncol(X)
    nonzeros = Y.nonzero()[0]
    try:
        pois_params, pois_cov = define_pois_params(Y[nonzeros]-1, X[nonzeros])
    except:
        pois_params = np.zeros(p)
        pois_cov = np.identity(p)

    bern_params, bern_cov = define_bern_params(Y, X)

    return bern_params, bern_cov, pois_params, pois_cov, p


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


    if isinstance(mod_list[0], dcmm):
        bern_list = [mod.bern_mod for mod in mod_list]
        pois_list = [mod.pois_mod for mod in mod_list]

        components = ['ntrend', 'nregn', 'nlf', 'nhol', 'seasPeriods', 'seasHarmComponents']
        kwargs = {name+'_bern': bern_list[0].__dict__.get(name) for name in components}
        kwargs.update({name+'_pois': pois_list[0].__dict__.get(name) for name in components})
        kwargs.update({'rho':pois_list[0].rho})
        kwargs.update({'nregn_bern': kwargs.get('nregn_bern') - kwargs.get('nhol_bern')})
        kwargs.update({'nregn_pois': kwargs.get('nregn_pois') - kwargs.get('nhol_pois')})

        # Discount factors averaged across models
        discount_factors = ['deltrend', 'delregn', 'delseas', 'dellf', 'delhol']
        kwargs.update({name + '_bern' : get_mean(bern_list, name) for name in discount_factors})
        kwargs.update({name + '_pois': get_mean(pois_list, name) for name in discount_factors})

        # Precision weighted average to get state vector mean
        kwargs.update({'a0_bern' : get_mean_pw(bern_list, 'a', 'R'),
                       'a0_pois' : get_mean_pw(pois_list, 'a', 'R')})

        # Define covariance matrix as the average across model list, inflated
        kwargs.update({'R0_bern' : get_mean(bern_list, 'R') * 5,
                       'R0_pois' : get_mean(pois_list, 'R') * 5})

        # Define model
        mod = dcmm(**kwargs)


    if isinstance(mod_list[0], dbcm):
        bern_list = [mod.dcmm.bern_mod for mod in mod_list]
        pois_list = [mod.dcmm.pois_mod for mod in mod_list]

        components = ['ntrend', 'nregn', 'nlf', 'nhol', 'seasPeriods', 'seasHarmComponents']
        kwargs = {name+'_bern': bern_list[0].__dict__.get(name) for name in components}
        kwargs.update({name+'_pois': pois_list[0].__dict__.get(name) for name in components})
        kwargs.update({name + '_cascade': mod_list[0].cascade[0].__dict__.get(name) for name in components})
        kwargs.update({'rho': pois_list[0].rho})
        kwargs.update({'nregn_bern': kwargs.get('nregn_bern') - kwargs.get('nhol_bern')})
        kwargs.update({'nregn_pois': kwargs.get('nregn_pois') - kwargs.get('nhol_pois')})
        kwargs.update({'nregn_cascade': kwargs.get('nregn_cascade') - kwargs.get('nhol_cascade')})

        # Discount factors averaged across models
        discount_factors = ['deltrend', 'delregn', 'delseas', 'dellf', 'delhol']
        kwargs.update({name + '_bern' : get_mean(bern_list, name) for name in discount_factors})
        kwargs.update({name + '_pois': get_mean(pois_list, name) for name in discount_factors})

        # Precision weighted average to get state vector mean
        kwargs.update({'a0_bern' : get_mean_pw(bern_list, 'a', 'R'),
                       'a0_pois' : get_mean_pw(pois_list, 'a', 'R')})
        a0_cascade = []
        for i in range(mod_list[0].ncascade):
            a0_cascade.append(get_mean_pw([mod.cascade[i] for mod in mod_list], 'a', 'R'))
        kwargs.update({'a0_cascade':a0_cascade})

        # Define covariance matrix as the average across model list, inflated
        kwargs.update({'R0_bern' : get_mean(bern_list, 'R') * 5,
                       'R0_pois' : get_mean(pois_list, 'R') * 5})
        R0_cascade = []
        for i in range(mod_list[0].ncascade):
            R0_cascade.append(get_mean([mod.cascade[i] for mod in mod_list], 'R'))
        kwargs.update({'R0_cascade': R0_cascade})

        mod = dbcm(**kwargs)

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