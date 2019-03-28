import numpy as np
import pandas as pd
from .dglm import normal_dlm
from .dcmm import dcmm
from .dbcm import dbcm
import statsmodels.api as sm

def define_normal_dlm(Y, prior_length, period=[7], harmComponents = [[1,2,3]], **kwargs):
    """
    :param Y: Observation array, length must be at least equal to or greater than the prior_length
    :param prior_length: Number of observations to be used in setting the prior
    :param period: List of periods for seasonal components
    :param harmComponents: List of harmonic components included in each seasonal component
    :return: Returns an initialized normal DLM
    """
    # Define normal DLM for total sales
    mean = Y[:prior_length].mean()
    a0 = np.array([[mean, 0, 0, 0, 0, 0, 0, 0, 0]])
    R0 = np.diag([.1, .1, .1, .5, .5, .5, .5, .5, .5])
    nmod = normal_dlm(a0 = a0, R0 = R0,
                nregn = 1,
                ntrend = 2,
                seasPeriods = period,
                seasHarmComponents = harmComponents,
                deltrend = .99, delregn = .995,
                delhol = 1, delseas = .995,
                n0 = 1, s0 = 1, delVar = .995)
    
    return nmod

def define_dcmm(Y, X, prior_length = 30, seasPeriods = [7], seasHarmComponents = [[1,2,3]], nmultiscale=0, rho=1,
                deltrend = .99, delregn =.99, delseas = .99, delmultiscale = .99,
                delbern = None, delpois = None, **kwargs):
    """
    :param Y: Observation array, must have length at least as long as prior_length
    :param X: Covariate array, must have length at least as long as prior_length
    :param prior_length: Number of observations to be used in setting the prior
    :param seasPeriods: List of periods for seasonal components
    :param seasHarmComponents: List of harmonic components included in each seasonal component
    :param nmultiscale: Number of multiscale components
    :param rho: Discount factor for random effects extension in poisson DGLM (smaller rho increases variance)
    :param deltrend: Discount factor on trend components in DCMM and cascade
    :param delregn: Discount factor on regression components in DCMM and cascade
    :param delseas: Discount factor on seasonal components in DCMM and cascade
    :param delmultiscale: Discount factor on multiscale components in DCMM and cascade
    :param delbern: Discount factor for all components of bernoulli DGLM
    :param delpois: Discount factor for all components of poisson DGLM
    :return: Returns an initialized DCMM
    """

    # suppport overwrite the defaults with regression and seasonal discounts
    nregn = ncol(X)
    ntrend = 1
    nseas = 2*sum(map(len, seasHarmComponents))

    deltrend_bern = deltrend if delbern is None else delbern
    deltrend_pois = deltrend if delpois is None else delpois

    delregn_bern = delregn if delbern is None else delbern
    delregn_pois = delregn if delpois is None else delpois

    delseas_bern = delseas if delbern is None else delbern
    delseas_pois = delseas if delpois is None else delpois

    delmultiscale_bern = delmultiscale if delbern is None else delbern
    delmultiscale_pois = delmultiscale if delpois is None else delpois

    pois_params, bern_params = define_dcmm_params(Y, X, prior_length)

    prior = [[*bern_params], [0] * nseas, [1] * nmultiscale]
    a0_bern = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    R0_bern = np.identity(a0_bern.shape[0])/2
    prior =[ [*pois_params], [0] * nseas, [1] * nmultiscale]
    a0_pois = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    R0_pois = np.identity(a0_pois.shape[0])/2
    mod = dcmm(a0_bern = a0_bern, R0_bern = R0_bern,
                nregn_bern = nregn,
                ntrend_bern = ntrend,
                nmultiscale_bern = nmultiscale,
                seasPeriods_bern = seasPeriods,
                seasHarmComponents_bern = seasHarmComponents,
                deltrend_bern = deltrend_bern, delregn_bern = delregn_bern,
                delseas_bern = delseas_bern,
                delmultiscale_bern=delmultiscale_bern,
          a0_pois = a0_pois, R0_pois = R0_pois,
                nregn_pois = nregn,
                ntrend_pois = ntrend,
                nmultiscale_pois=nmultiscale,
                seasPeriods_pois = seasPeriods,
                seasHarmComponents_pois = seasHarmComponents,
                deltrend_pois = deltrend_pois, delregn_pois = delregn_pois,
                delseas_pois = delseas_pois,
                delmultiscale_pois=delmultiscale_pois,
               rho = rho)
        
    return mod


def define_dbcm(Y_transaction, X_transaction = None, Y_cascade = None, X_cascade = None, excess_baskets = [], excess_values=[],
                prior_length = 30, seasPeriods = [7], seasHarmComponents = [[1,2,3]], nmultiscale=0,
                rho = 1, deltrend = .99, delregn =.99, delseas = .99, delmultiscale = .99,
                delbern = None, delpois = None, delcascade = None, **kwargs):
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
    :param nmultiscale: Number of multiscale components
    :param rho: Discount factor for random effects extension in poisson DGLM (smaller rho increases variance)
    :param deltrend: Discount factor on trend components in DCMM and cascade
    :param delregn: Discount factor on regression components in DCMM and cascade
    :param delseas: Discount factor on seasonal components in DCMM and cascade
    :param delmultiscale: Discount factor on multiscale components in DCMM and cascade
    :param delbern: Discount factor for all components of bernoulli DGLM
    :param delpois: Discount factor for all components of poisson DGLM
    :return: Returns an initialized DBCM
    """

    nregn = ncol(X_transaction)
    ntrend = 1
    nseas = 2 * sum(map(len, seasHarmComponents))

    deltrend_bern = deltrend if delbern is None else delbern
    deltrend_pois = deltrend if delpois is None else delpois
    deltrend_cascade = deltrend if delcascade is None else delcascade

    delregn_bern = delregn if delbern is None else delbern
    delregn_pois = delregn if delpois is None else delpois
    delregn_cascade = delregn if delcascade is None else delcascade

    delseas_bern = delseas if delbern is None else delbern
    delseas_pois = delseas if delpois is None else delpois
    delseas_cascade = delseas if delcascade is None else delcascade

    delmultiscale_bern = delmultiscale if delbern is None else delbern
    delmultiscale_pois = delmultiscale if delpois is None else delpois
    delmultiscale_cascade = delmultiscale if delcascade is None else delcascade

    # Fit a GLM for the poisson and bernoulli components of the DCMM on transactions
    pois_params, bern_params = define_dcmm_params(Y_transaction, X_transaction, prior_length)

    prior = [[*bern_params], [0] * nseas, [1] * nmultiscale]
    a0_bern = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    R0_bern = np.identity(a0_bern.shape[0]) / 2
    prior = [[*pois_params], [0] * nseas, [1] * nmultiscale]
    a0_pois = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    R0_pois = np.identity(a0_pois.shape[0]) / 2

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
                nmultiscale_bern=nmultiscale,
                seasPeriods_bern = seasPeriods,
                seasHarmComponents_bern = seasHarmComponents,
                deltrend_bern = deltrend_bern, delregn_bern = delregn_bern, delseas_bern = delseas_bern,
                delmultiscale_bern = delmultiscale_bern,

          a0_pois = a0_pois, R0_pois = R0_pois,
                nregn_pois = nregn,
                ntrend_pois = ntrend,
                nmultiscale_pois=nmultiscale,
                seasPeriods_pois = seasPeriods,
                seasHarmComponents_pois = seasHarmComponents,
                deltrend_pois = deltrend_pois, delregn_pois = delregn_pois, delseas_pois = delseas_pois,
                delmultiscale_pois = delmultiscale_pois, rho = rho,

          ncascade = ncascade,
             a0_cascade = a0_cascade, # List of length ncascade
             R0_cascade = R0_cascade, # List of length ncascade
             nregn_cascade = nregn_cascade,
             ntrend_cascade = 1,
             nmultiscale_cascade = 0,
             seasPeriods_cascade = [],
             seasHarmComponents_cascade = [],
             deltrend_cascade = deltrend_cascade, delregn_cascade = delregn_cascade, delseas_cascade = delseas_cascade,

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
    pois_mod = sm.GLM(Y[nonzeros] - 1,
                      np.c_[np.ones([len(nonzeros), 1]), X[nonzeros]],
                      family=sm.families.Poisson())
    pois_params = pois_mod.fit().params

    if len(nonzeros) + 4 >= prior_length or len(nonzeros) <= 4:
        bernmean = len(nonzeros) / (prior_length + 1)
        bernmean = np.log(bernmean / (1 - bernmean))
        bern_params = np.zeros(pois_params.shape)
        bern_params[0] = bernmean
    else:
        Y_bern = np.c_[np.zeros([prior_length, 1]), np.ones([prior_length, 1])]
        Y_bern[Y[:prior_length].nonzero()[0], 0] = 1
        Y_bern[Y[:prior_length].nonzero()[0], 1] = 0
        X_bern = np.c_[np.ones([prior_length, 1]), X[:prior_length]]
        bern_mod = sm.GLM(endog=Y_bern, exog=X_bern, family=sm.families.Binomial())
        bern_params = bern_mod.fit().params

    return pois_params, bern_params

def ncol(x):
    if len(np.shape(x)) == 1:
        return 1
    else:
        return np.shape(x)[1]