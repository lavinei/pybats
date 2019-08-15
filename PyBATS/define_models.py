import numpy as np
import pandas as pd
from .dglm import normal_dlm
from .dcmm import dcmm
from .dbcm import dbcm
from .shared import define_holiday_regressors
import statsmodels.api as sm
from pandas.tseries.holiday import AbstractHolidayCalendar

def define_normal_dlm(Y, X, prior_length, ntrend=2, nlf=0, nhol=0, seasPeriods=[7], seasHarmComponents = [[1, 2, 3]],
                      deltrend = .995, delregn =.995, delseas = .999, dellf=.999, delVar = 0.999, delhol=1,
                      n0 = 1, s0 = 1, a0=None, R0=None,
                      adapt_discount=False, **kwargs):
    """
    :param Y: Observation array, length must be at least equal to or greater than the prior_length
    :param prior_length: Number of observations to be used in setting the prior
    :param seasPeriods: List of periods for seasonal components
    :param seasHarmComponents: List of harmonic components included in each seasonal component
    :return: Returns an initialized normal DLM
    """
    # Define normal DLM for total sales
    nregn = ncol(X) - nhol
    nseas = 2 * sum(map(len, seasHarmComponents))
    if prior_length > 0:
        if X is None:
            prior_OLS = sm.OLS(Y[:prior_length], np.ones([prior_length, 1])).fit()
        else:
            prior_OLS = sm.OLS(Y[:prior_length], sm.add_constant(X[:prior_length, :])).fit()
        dlm_params_mean = prior_OLS.params
        dlm_params_cov = prior_OLS.cov_params().diagonal()

        prior = [[dlm_params_mean[0]], [0] * (ntrend-1), [*dlm_params_mean[1:]], [0] * nseas, [1] * nlf]
    if a0 is None:
        a0 = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    # prior_cov = [[dlm_params_cov[0]], [1]*(ntrend-1), [*dlm_params_cov[1:]], [1] * nseas, [1]*nlf]
    # R0 = np.diag([np.max(1, v) for vs in prior_cov for v in vs])
    if R0 is None:
        R0 = np.identity(a0.shape[0])*2
    nmod = normal_dlm(a0 = a0, R0 = R0,
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
    
    return nmod

def define_dcmm(Y, X, prior_length = 30, seasPeriods = [7], seasHarmComponents = [[1,2,3]], nlf=0, rho=1, nhol = 0,
                deltrend_bern=.995, delregn_bern=.995, delseas_bern=.995, dellf_bern=.999, delhol_bern=1,
                deltrend_pois=.998, delregn_pois=.995, delseas_pois=.995, dellf_pois=.999, delhol_pois=1,
                a0_bern = None, R0_bern = None, a0_pois = None, R0_pois = None,
                interpolate=True, adapt_discount=False,
                **kwargs):
    """
    :param Y: Observation array, must have length at least as long as prior_length
    :param X: Covariate array, must have length at least as long as prior_length
    :param prior_length: Number of observations to be used in setting the prior
    :param seasPeriods: List of periods for seasonal components
    :param seasHarmComponents: List of harmonic components included in each seasonal component
    :param nlf: Number of latent factor components
    :param rho: Discount factor for random effects extension in poisson DGLM (smaller rho increases variance)
    :param nhol: Number of holidays to include in the model
    :param deltrend: Discount factor on trend components in DCMM and cascade
    :param delregn: Discount factor on regression components in DCMM and cascade
    :param delseas: Discount factor on seasonal components in DCMM and cascade
    :param dellf: Discount factor on latent factor components in DCMM and cascade
    :param delbern: Discount factor for all components of bernoulli DGLM
    :param delpois: Discount factor for all components of poisson DGLM
    :param adapt_discount: Can be 'info' or 'positive_regn' as 2 ways to adapt discount factors and prevent variance blowing up
    :return: Returns an initialized DCMM
    """

    # suppport overwrite the defaults with regression and seasonal discounts
    nregn = ncol(X) - nhol
    ntrend = 1
    nseas = 2*sum(map(len, seasHarmComponents))

    pois_params, bern_params = define_dcmm_params(Y, X, prior_length)

    #print(pois_params, bern_params)

    prior = [[*bern_params], [0] * nseas, [1] * nlf]
    if a0_bern is None: a0_bern = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    if R0_bern is None: R0_bern = np.identity(a0_bern.shape[0])

    prior =[[*pois_params], [0] * nseas, [1] * nlf]
    if a0_pois is None: a0_pois = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    if R0_pois is None: R0_pois = np.identity(a0_pois.shape[0])

    # Double the variance on holiday indicators
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


def define_dbcm(Y_transaction, X_transaction = None, Y_cascade = None, X_cascade = None, excess_baskets = [], excess_values=[],
                prior_length = 30, seasPeriods = [7], seasHarmComponents = [[1,2,3]], nlf=0,
                rho = 1, nhol = 0,
                deltrend_bern = .995, delregn_bern =.995, delseas_bern = .995, dellf_bern = .999, delhol_bern = 1,
                deltrend_pois = .998, delregn_pois =.995, delseas_pois = .995, dellf_pois = .999, delhol_pois = 1,
                deltrend_cascade = .999, delregn_cascade =1., delseas_cascade = .999, dellf_cascade = .999, delhol_cascade = 1.,
                a0_bern = None, R0_bern = None, a0_pois = None, R0_pois=None,
                interpolate=True, adapt_discount=False,
                **kwargs):
    """
    :param Y_transaction: Observation array of transactions, must have length at least as long as prior_length
    :param X_transaction: Covariate array associated with the transactions (used in the DCMM)
    :param Y_cascade: Observation array of basket sizes, must have length at least as long as prior_length
    :param X_cascade: Covariate array associated with the binary cascade on basket sizes
    :param excess_baskets: Either excess_baskets or excess_values must be defined. Excess baskets is an array of basket sizes
    in exactly the same format as the Y_cascade values. Column j indicate number of transactions with more than j items.
    :param excess_values: List of excess values. Each element gives excess basket sizes observed on that day.
    Most days should be an empty list. Some days may have multiple transactions with excess basket sizes.
    :param prior_length: Number of observations to be used in setting the prior
    :param seasPeriods: List of periods for seasonal components
    :param seasHarmComponents: List of harmonic components included in each seasonal component
    :param nlf: Number of latent factor components
    :param rho: Discount factor for random effects extension in poisson DGLM (smaller rho increases variance)
    :param deltrend: Discount factor on trend components in DCMM and cascade
    :param delregn: Discount factor on regression components in DCMM and cascade
    :param delseas: Discount factor on seasonal components in DCMM and cascade
    :param dellf: Discount factor on latent factor components in DCMM and cascade
    :param delbern: Discount factor for all components of bernoulli DGLM
    :param delpois: Discount factor for all components of poisson DGLM
    :param adapt_discount: Can be 'info' or 'positive_regn' as 2 ways to adapt discount factors and prevent variance blowing up
    :return: Returns an initialized DBCM
    """

    nregn = ncol(X_transaction) - nhol
    ntrend = 1
    nseas = 2 * sum(map(len, seasHarmComponents))

    # Fit a GLM for the poisson and bernoulli components of the DCMM on transactions
    pois_params, bern_params = define_dcmm_params(Y_transaction, X_transaction, prior_length)

    prior = [[*bern_params], [0] * nseas, [1] * nlf]
    if a0_bern is None: a0_bern = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    if R0_bern is None: R0_bern = np.identity(a0_bern.shape[0])/2
    prior = [[*pois_params], [0] * nseas, [1] * nlf]
    if a0_pois is None: a0_pois = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    if R0_pois is None: R0_pois = np.identity(a0_pois.shape[0])

    # Double the variance on holiday indicators
    ihol = range(ntrend + nregn, ntrend + nregn + nhol)
    for idx in ihol:
        R0_bern[idx, idx] = R0_bern[idx, idx] * 4
        R0_pois[idx, idx] = R0_pois[idx, idx] * 4

    # Calculate the prior means for the Cascade
    def cascade_prior_mean(alpha, beta):
        alpha += 1
        beta += 1
        mean = alpha / (alpha + beta)
        logit_mean = np.log(mean / (1 - mean))
        return logit_mean

    # Calculate the prior means for the cascades
    ncascade = Y_cascade.shape[1]
    nregn_cascade = ncol(X_cascade)
    ntrend_cascade = 1
    pcascade = nregn_cascade + ntrend_cascade

    Yc = np.c_[Y_transaction, Y_cascade]
    Yc = np.sum(Yc[:prior_length], axis = 0)
    means = [cascade_prior_mean(Yc[i+1], Yc[i] - Yc[i+1]) for i in range(ncascade)]
    a0_cascade = [np.zeros(pcascade).reshape(-1,1) for i in range(ncascade)]
    for i, m in enumerate(means):
        a0_cascade[i][0] = m
    R0_cascade = [0.1 * np.identity(pcascade) for i in range(ncascade)]

    # Initialize empirically observed excess baskets
    excess = []
    if len(excess_values) == 0 and len(excess_baskets) > 0:
            counts = np.sum(excess_baskets[:prior_length, :], axis=0)
            counts[:len(counts)-1] = counts[:len(counts)-1] - counts[1:]
            for val, count in enumerate(counts):
                excess.extend([val + ncascade + 1 for c in range(count)])
    else:
        for e in excess_values[:prior_length]:
            excess.extend(e)

    # Define the model
    mod = dbcm(a0_bern = a0_bern, R0_bern = R0_bern,
               nregn_bern = nregn,
               ntrend_bern = ntrend,
               nlf_bern=nlf,
               nhol_bern = nhol,
               seasPeriods_bern = seasPeriods,
               seasHarmComponents_bern = seasHarmComponents,
               deltrend_bern = deltrend_bern, delregn_bern = delregn_bern, delseas_bern = delseas_bern,
               dellf_bern= dellf_bern, delhol_bern = delhol_bern,

               a0_pois = a0_pois, R0_pois = R0_pois,
               nregn_pois = nregn,
               ntrend_pois = ntrend,
               nlf_pois=nlf,
               nhol_pois=nhol,
               seasPeriods_pois = seasPeriods,
               seasHarmComponents_pois = seasHarmComponents,
               deltrend_pois = deltrend_pois, delregn_pois = delregn_pois, delseas_pois = delseas_pois,
               dellf_pois= dellf_pois, delhol_pois = delhol_pois, rho = rho,
               interpolate=interpolate,
               adapt_discount=adapt_discount,

               ncascade = ncascade,
               a0_cascade = a0_cascade,  # List of length ncascade
               R0_cascade = R0_cascade,  # List of length ncascade
               nregn_cascade = nregn_cascade,
               ntrend_cascade = 1,
               nlf_cascade= 0,
               seasPeriods_cascade = [],
               seasHarmComponents_cascade = [],
               deltrend_cascade = deltrend_cascade, delregn_cascade = delregn_cascade,
               excess = excess)
        
    return mod

def define_dcmm_params(Y, X, prior_length):
    """
    Helper function to initialize parameters for the Bernoulli and Poisson DGLMs within a DCMM
    :param Y:
    :param X:
    :param prior_length:
    :return:
    """
    nonzeros = Y[:prior_length].nonzero()[0]
    # if Y[:prior_length].max() > 1.0:
    try:
        pois_mod = sm.GLM(Y[nonzeros] - 1,
                          np.c_[np.ones([len(nonzeros), 1]), X[nonzeros]],
                          family=sm.families.Poisson())
        pois_params = pois_mod.fit().params
    except:
        pois_params = np.zeros(ncol(X) + 1)
    # else:
    #     pois_params = np.zeros(ncol(X) + 1)

    # if len(nonzeros) + 4 >= prior_length or len(nonzeros) <= 4:
    #     bernmean = len(nonzeros) / (prior_length + 1)
    #     bernmean = np.log(bernmean / (1 - bernmean))
    #     bern_params = np.zeros(pois_params.shape)
    #     bern_params[0] = bernmean
    # else:
    try:
        Y_bern = np.c_[np.zeros([prior_length, 1]), np.ones([prior_length, 1])]
        Y_bern[Y[:prior_length].nonzero()[0], 0] = 1
        Y_bern[Y[:prior_length].nonzero()[0], 1] = 0
        X_bern = np.c_[np.ones([prior_length, 1]), X[:prior_length]]
        bern_mod = sm.GLM(endog=Y_bern, exog=X_bern, family=sm.families.Binomial())
        bern_params = bern_mod.fit().params
    except:
        if len(nonzeros) > 0:
            bernmean = len(nonzeros) / (prior_length + 1)
            bernmean = np.log(bernmean / (1 - bernmean))
            bern_params = np.zeros(pois_params.shape)
            bern_params[0] = bernmean
        else:
            bern_params = np.zeros(pois_params.shape)

    return pois_params, bern_params

def ncol(x):
    if x is None:
        return 0
    if len(np.shape(x)) == 1:
        return 1
    else:
        return np.shape(x)[1]