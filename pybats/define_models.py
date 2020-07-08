import numpy as np

from .dbcm import dbcm
from .dcmm import dcmm

from .dglm import dlm, pois_dglm, bern_dglm, bin_dglm
import statsmodels.api as sm


def define_dglm(Y, X, family="normal", n=None,
                ntrend=1, nlf=0, nhol=0,
                seasPeriods=[7], seasHarmComponents = [[1, 2, 3]],
                deltrend = .995, delregn =.995, delseas = .999, dellf=.999, delVar = 0.999, delhol=1,
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

        # Selecting only the correct number of observations (relevant if prior_length is given)
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
    else:
        # Infer the number of regression and holiday components
        p = len(a0)
        nseas = 2 * sum(map(len, seasHarmComponents))
        nregn = p - ntrend - nhol - nseas - nlf


    if return_aR:
        return a0, R0, nregn

    if kwargs.get('rho') is not None:
        rho = kwargs.get('rho')
    else:
        rho = 1

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
                  adapt_discount=adapt_discount,
                  discount_forecast = discount_forecast)
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
                        adapt_discount=adapt_discount,
                        discount_forecast = discount_forecast,
                        rho = rho)
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
                        adapt_discount=adapt_discount,
                        discount_forecast = discount_forecast,
                        rho=rho)
    elif family == "binomial":
        mod = bin_dglm(a0=a0, R0=R0,
                       nregn=nregn,
                       ntrend=ntrend,
                       nlf=nlf,
                       nhol=nhol,
                       seasPeriods=seasPeriods,
                       seasHarmComponents=seasHarmComponents,
                       deltrend=deltrend, delregn=delregn,
                       delseas=delseas, delhol=delhol,
                       dellf=dellf,
                       adapt_discount=adapt_discount,
                       discount_forecast = discount_forecast,
                       rho=rho)


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

    nonzeros = Y.nonzero()[0]
    pois_mod = define_dglm(Y[nonzeros] - 1, X[nonzeros], family="poisson", ntrend=ntrend, nlf=nlf, nhol=nhol,
                              seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                              a0=a0_pois, R0=R0_pois, prior_length=prior_length)
    bern_mod = define_dglm(Y, X, family="bernoulli", ntrend=ntrend, nlf=nlf, nhol=nhol,
                              seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                              a0=a0_bern, R0=R0_bern, prior_length=prior_length)



    mod = dcmm(a0_bern = bern_mod.a, R0_bern = bern_mod.R,
               nregn_bern = bern_mod.nregn_exhol,
               ntrend_bern = bern_mod.ntrend,
               nlf_bern= bern_mod.nlf,
               nhol_bern=bern_mod.nhol,
               seasPeriods_bern = bern_mod.seasPeriods,
               seasHarmComponents_bern = bern_mod.seasHarmComponents,
               deltrend_bern = deltrend_bern, delregn_bern = delregn_bern,
               delseas_bern = delseas_bern,
               dellf_bern=dellf_bern,
               delhol_bern = delhol_bern,
               a0_pois = pois_mod.a, R0_pois = pois_mod.R,
               nregn_pois = pois_mod.nregn_exhol,
               ntrend_pois = pois_mod.ntrend,
               nlf_pois=pois_mod.nlf,
               nhol_pois=pois_mod.nhol,
               seasPeriods_pois = pois_mod.seasPeriods,
               seasHarmComponents_pois = pois_mod.seasHarmComponents,
               deltrend_pois = deltrend_pois, delregn_pois = delregn_pois,
               delseas_pois = delseas_pois,
               dellf_pois=dellf_pois,
               delhol_pois = delhol_pois,
               rho = rho,
               interpolate=interpolate,
               adapt_discount=adapt_discount
               )


    # if prior_length is not None:
    #     if prior_length > 0:
    #         Y = Y[:prior_length]
    #         X = X[:prior_length]
    #
    # nregn = ncol(X) - nhol
    # nseas = 2*sum(map(len, seasHarmComponents))
    #
    # bern_params, bern_cov, pois_params, pois_cov, p = define_dcmm_params(Y, X)
    #
    # prior = [[bern_params[0]], [0] * (ntrend-1), [*bern_params[1:]], [0] * nseas, [1] * nlf]
    # if a0_bern is None: a0_bern = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    # if R0_bern is None:
    #     R0_bern = np.identity(a0_bern.shape[0])
    #     idx = [i for i in range(p + ntrend - 1)]
    #     for j in range(1, ntrend):
    #         idx.pop(j)
    #     R0_bern[np.ix_(idx, idx)] = bern_cov
    #
    # prior = [[pois_params[0]], [0] * (ntrend-1), [*pois_params[1:]], [0] * nseas, [1] * nlf]
    # if a0_pois is None: a0_pois = np.array([m for ms in prior for m in ms]).reshape(-1, 1)
    # if R0_pois is None:
    #     R0_pois = np.identity(a0_pois.shape[0])
    #     idx = [i for i in range(p + ntrend - 1)]
    #     for j in range(1, ntrend):
    #         idx.pop(j)
    #     R0_pois[np.ix_(idx, idx)] = pois_cov
    #
    # # Add variance to holiday indicators
    # ihol = range(ntrend + nregn, ntrend + nregn + nhol)
    # for idx in ihol:
    #     R0_bern[idx, idx] = R0_bern[idx, idx] * 4
    #     R0_pois[idx, idx] = R0_pois[idx, idx] * 4
    #
    #
    # mod = dcmm(a0_bern = a0_bern, R0_bern = R0_bern,
    #            nregn_bern = nregn,
    #            ntrend_bern = ntrend,
    #            nlf_bern= nlf,
    #            nhol_bern=nhol,
    #            seasPeriods_bern = seasPeriods,
    #            seasHarmComponents_bern = seasHarmComponents,
    #            deltrend_bern = deltrend_bern, delregn_bern = delregn_bern,
    #            delseas_bern = delseas_bern,
    #            dellf_bern=dellf_bern,
    #            delhol_bern = delhol_bern,
    #            a0_pois = a0_pois, R0_pois = R0_pois,
    #            nregn_pois = nregn,
    #            ntrend_pois = ntrend,
    #            nlf_pois=nlf,
    #            nhol_pois=nhol,
    #            seasPeriods_pois = seasPeriods,
    #            seasHarmComponents_pois = seasHarmComponents,
    #            deltrend_pois = deltrend_pois, delregn_pois = delregn_pois,
    #            delseas_pois = delseas_pois,
    #            dellf_pois=dellf_pois,
    #            delhol_pois = delhol_pois,
    #            rho = rho,
    #            interpolate=interpolate,
    #            adapt_discount=adapt_discount
    #            )

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
            nonan = ~np.any(np.isnan(Yc), axis=1)
            Yc = np.sum(Yc[:prior_length][nonan[:prior_length]], axis=0)
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