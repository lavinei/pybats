import numpy as np
from forecasting.dglm import normal_dlm
from forecasting.dcmm import dcmm
from forecasting.dbcm import dbcm
import statsmodels.api as sm

def define_models(Y, Y_totalsales, prior_length = 30):
    nmod = define_normal_dlm(Y_totalsales, prior_length)
    dcmm_standard = define_dcmm(Y, prior_length)
    dcmm_multiscale = define_dcmm(Y, prior_length, multiscale = True)
    return nmod, dcmm_standard, dcmm_multiscale

def define_normal_dlm(Y, prior_length):
    # Define normal DLM for total sales
    mean = Y[:prior_length].mean()
    a0 = np.array([[mean, 0, 0, 0, 0, 0, 0, 0]])
    R0 = np.diag([.1, .1, .5, .5, .5, .5, .5, .5])
    nmod = normal_dlm(a0 = a0, R0 = R0,
                nregn = 1,
                ntrend = 1,
                seasPeriods = [7],
                seasHarmComponents = [[1,2,3]],
                deltrend = .99, delregn = .995,
                delhol = 1, delseas = .995,
                n0 = 1, s0 = 1, delVar = .995)
    
    return nmod

def define_dcmm(Y, prior_length = 30, seasPeriods = [7], seasHarmComponents = [[1,2,3]], multiscale = False, rho=1, delbern = .999, delpois =.995):
    bernmean = len(Y[:prior_length].nonzero()[0])/(prior_length+1)
    bernmean = np.log(bernmean / (1-bernmean))
    poismean = np.log(Y[:prior_length].mean())
    if not multiscale:
        # Define a standard DCMM for a single item's sales (as a comparison)
        a0_bern = np.array([[bernmean, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
        R0_bern = np.identity(8)/2
        a0_pois = np.array([[poismean, 0, 0, 0, 0, 0, 0, 0]])
        R0_pois = np.diag([.3, .3, .5, .5, .5, .5, .5, .5])
        mod = dcmm(a0_bern = a0_bern, R0_bern = R0_bern,
                    nregn_bern = 1,
                    ntrend_bern = 1,
                    seasPeriods_bern = seasPeriods,
                    seasHarmComponents_bern = seasHarmComponents,
                    deltrend_bern = delbern, delregn_bern = delbern,
                    delhol_bern = delbern, delseas_bern = delbern,
              a0_pois = a0_pois, R0_pois = R0_pois,
                    nregn_pois = 1,
                    ntrend_pois = 1,
                    seasPeriods_pois = seasPeriods,
                    seasHarmComponents_pois = seasHarmComponents,
                    deltrend_pois = delpois, delregn_pois = delpois,
                    delhol_pois = delpois, delseas_pois = delpois,
                   rho = rho)
    elif multiscale:
        # Define a multiscale DCMM for that single item's sales
        a0_bern = np.array([[bernmean, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
        R0_bern = np.identity(9)/2
        a0_pois = np.array([[poismean, 0, 0, 0, 0, 0, 0, 0, 0]])
        R0_pois = np.diag([.3, .3, .5, .5, .5, .5, .5, .5, .5])
        mod = dcmm(a0_bern = a0_bern, R0_bern = R0_bern,
                    nregn_bern = 1,
                    ntrend_bern = 1,
                    nmultiscale_bern = 7,
                    deltrend_bern = delbern, delregn_bern = delbern,
                    delhol_bern = delbern, delseas_bern = delbern,
                    delmultiscale_bern = delbern,
              a0_pois = a0_pois, R0_pois = R0_pois,
                    nregn_pois = 1,
                    ntrend_pois = 1,
                    nmultiscale_pois = 7,
                    deltrend_pois = delpois, delregn_pois = delpois,
                    delhol_pois = delpois, delseas_pois = delpois,
                    delmultiscale_pois = delpois,
                   rho=rho)
        
    return mod


def define_dbcm(Y_transaction, X_transaction = None, Y_cascade = None, X_cascade = None, excess_baskets = [], excess_values=[],
                prior_length = 30, seasPeriods = [7], seasHarmComponents = [[1,2,3]], multiscale = False,
                rho = 1, deltrend = .99, delregn =.99, delseas = .99, delmultiscale = .99, delhol=1):
    def ncol(x):
        if len(np.shape(x)) == 1:
            return 1
        else:
            return np.shape(x)[1]

    # Fit a GLM for the poisson and bernoulli components of the DCMM on transactions
    nonzeros = Y_transaction[:prior_length].nonzero()[0]
    pois_mod = sm.GLM(Y_transaction[nonzeros] - 1,
                      np.c_[np.ones([len(nonzeros), 1]), X_transaction[nonzeros]],
                      family=sm.families.Poisson())
    pois_params = pois_mod.fit().params

    if len(nonzeros) + 4 >= prior_length or len(nonzeros) <= 4:
        bernmean = len(nonzeros) / (prior_length + 1)
        bernmean = np.log(bernmean / (1 - bernmean))
        bern_params = np.array([bernmean, 0])
    else:
        Y_bern = np.c_[np.zeros([prior_length, 1]), np.ones([prior_length, 1])]
        Y_bern[Y_transaction[:prior_length].nonzero()[0], 0] = 1
        Y_bern[Y_transaction[:prior_length].nonzero()[0], 1] = 0
        X_bern = np.c_[np.ones([prior_length, 1]), X_transaction[:prior_length]]
        bern_mod = sm.GLM(endog = Y_bern, exog = X_bern, family=sm.families.Binomial())
        bern_params = bern_mod.fit().params

    # Calculate the prior means for the Cascade
    def cascade_prior_mean(alpha, beta):
        alpha += 1
        beta += 1
        mean = alpha / (alpha + beta)
        logit_mean = np.log(mean / (1 - mean))
        return logit_mean

    # Calculate the prior means for the cascades
    ncascade = Y_cascade.shape[1]
    Yc = np.c_[Y_transaction - Y_cascade[:,0], Y_cascade]
    Yc = np.sum(Yc[:prior_length], axis = 0)
    pcascade = 1 + ncol(X_cascade)
    means = [cascade_prior_mean(Yc[i+1], Yc[i]) for i in range(ncascade)]
    a0_cascade = [np.array([m, 0]).reshape(-1,1) for m in means]
    R0_cascade = [np.array([[.1, 0], [0, 0.1]]) for i in range(ncascade)]

    # Gather the prior excess basket sizes observed
    excess = []
    if len(excess_values) == 0 and len(excess_baskets) > 0:
            counts = np.sum(excess_baskets[:prior_length, :], axis=0)
            counts[:len(counts)-1] = counts[:len(counts)-1] - counts[1:]
            for val, count in enumerate(counts):
                excess.extend([val + ncascade + 1 for c in range(count)])
    else:
        for e in excess_values[:prior_length]:
            excess.extend(e)

    
    if not multiscale:
        a0_bern = np.array([[bern_params[0], bern_params[1], 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
        R0_bern = np.identity(8)/2
        a0_pois = np.array([[pois_params[0], pois_params[1], 0, 0, 0, 0, 0, 0]])
        R0_pois = np.identity(8)
        mod = dbcm(a0_bern = a0_bern, R0_bern = R0_bern,
                    nregn_bern = 1,
                    ntrend_bern = 1,
                    seasPeriods_bern = seasPeriods,
                    seasHarmComponents_bern = seasHarmComponents,
                    deltrend_bern = deltrend, delregn_bern = delregn,
                    delhol_bern = delhol, delseas_bern = delseas,
                   
              a0_pois = a0_pois, R0_pois = R0_pois,
                    nregn_pois = 1,
                    ntrend_pois = 1,
                    seasPeriods_pois = seasPeriods,
                    seasHarmComponents_pois = seasHarmComponents,
                    deltrend_pois = deltrend, delregn_pois = delregn,
                    delhol_pois = delhol, delseas_pois = delseas,
                    rho = rho,
              
              ncascade = ncascade,
                 a0_cascade = a0_cascade, # List of length ncascade
                 R0_cascade = R0_cascade, # List of length ncascade
                 nregn_cascade = 1,
                 ntrend_cascade = 1,
                 nmultiscale_cascade = 0,
                 seasPeriods_cascade = None,
                 seasHarmComponents_cascade = None,
                 deltrend_cascade = deltrend, delregn_cascade = delregn,
                 delhol_cascade = delhol, delseas_cascade = delseas,

                excess = excess)
        
    elif multiscale:
        a0_bern = np.array([[bern_params[0], bern_params[1], 1, 1, 1, 1, 1, 1, 1]]).reshape(-1, 1)
        R0_bern = np.identity(9)
        a0_pois = np.array([[pois_params[0], pois_params[1], 1, 1, 1, 1, 1, 1, 1]])
        R0_pois = np.identity(9)
        mod = dbcm(a0_bern = a0_bern, R0_bern = R0_bern,
                    nregn_bern = 1,
                    ntrend_bern = 1,
                    nmultiscale_bern = 7,
                    deltrend_bern = deltrend, delregn_bern = delregn,
                    delhol_bern = delhol, delseas_bern = delseas,
                    delmultiscale_bern = delmultiscale,

              a0_pois = a0_pois, R0_pois = R0_pois,
                    nregn_pois = 1,
                    ntrend_pois = 1,
                    nmultiscale_pois = 7,
                    deltrend_pois = deltrend, delregn_pois = delregn,
                    delhol_pois = delhol, delseas_pois = delseas,
                    delmultiscale_pois = delmultiscale,
                    rho=rho,
                  
               ncascade = ncascade,
                 a0_cascade = a0_cascade, # List of length ncascade
                 R0_cascade = R0_cascade, # List of length ncascade
                 nregn_cascade = 1,
                 ntrend_cascade = 1,
                 nmultiscale_cascade = 0, # No multiscale cascades, only the DCMM portion is multiscale
                 seasPeriods_cascade = None,
                 seasHarmComponents_cascade = None,
                 deltrend_cascade = deltrend, delregn_cascade = delregn,
                 delhol_cascade = delhol, delseas_cascade = delseas,
                 delmultiscale_cascade = delmultiscale,

                excess = excess)
        
    return mod

def define_models_old():
    # Define a standard DCMM for a single item's sales (as a comparison)
    a0_bern = np.array([[1, 1, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
    R0_bern = np.identity(8)/2
    a0_pois = np.array([[.5, 0, 0, 0, 0, 0, 0, 0]])
    R0_pois = np.diag([.1, .1, .5, .5, .5, .5, .5, .5])
    dcmm_standard = dcmm(a0_bern = a0_bern, R0_bern = R0_bern,
                nregn_bern = 1,
                ntrend_bern = 1,
                seasPeriods_bern = [7],
                seasHarmComponents_bern = [[1,2,3]],
                deltrend_bern = .99, delregn_bern = .995,
                delhol_bern = 1, delseas_bern = .995,
          a0_pois = a0_pois, R0_pois = R0_pois,
                nregn_pois = 1,
                ntrend_pois = 1,
                seasPeriods_pois = [7],
                seasHarmComponents_pois = [[1,2,3]],
                deltrend_pois = .99, delregn_pois = .995,
                delhol_pois = 1, delseas_pois = .995)
    
    # Define a multiscale DCMM for that single item's sales
    a0_bern = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
    R0_bern = np.identity(9)/2
    a0_pois = np.array([[.5, 0, 0, 0, 0, 0, 0, 0, 0]])
    R0_pois = np.diag([.1, .1, .5, .5, .5, .5, .5, .5, .5])
    dcmm_multiscale = dcmm(a0_bern = a0_bern, R0_bern = R0_bern,
                nregn_bern = 1,
                ntrend_bern = 1,
                nmultiscale_bern = 7,
                deltrend_bern = .99, delregn_bern = .995,
                delhol_bern = 1, delseas_bern = .995,
                delmultiscale_bern = .995,
          a0_pois = a0_pois, R0_pois = R0_pois,
                nregn_pois = 1,
                ntrend_pois = 1,
                nmultiscale_pois = 7,
                deltrend_pois = .99, delregn_pois = .995,
                delhol_pois = 1, delseas_pois = .995,
                delmultiscale_pois = .995,)
    
    # Define normal DLM for total sales
    a0 = np.array([[1.9, 1.5, 0, 0, 0, 0, 0, 0]])
    R0 = np.diag([.1, .1, .5, .5, .5, .5, .5, .5])
    nmod = normal_dlm(a0 = a0, R0 = R0,
                nregn = 1,
                ntrend = 1,
                seasPeriods = [7],
                seasHarmComponents = [[1,2,3]],
                deltrend = .99, delregn = .995,
                delhol = 1, delseas = .995,
                n0 = 1, s0 = 1, delVar = .995)
    
    return nmod, dcmm_standard, dcmm_multiscale