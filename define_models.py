import numpy as np
from dglm import *
import seaborn as sns
from forecasting.forecast import *
from forecasting.multiscale import *
from forecasting.seasonal import *
from forecasting.update import *
from forecasting.dcmm import *
from forecasting.dbcm import *

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

def define_dcmm(Y, prior_length = 30, seasPeriods = [7], seasHarmComponents = [[1,2,3]], multiscale = False):
    bernmean = len(Y[:prior_length].nonzero()[0])/prior_length
    poismean = Y[:prior_length].mean()
    if not multiscale:
        # Define a standard DCMM for a single item's sales (as a comparison)
        a0_bern = np.array([[bernmean, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
        R0_bern = np.identity(8)/2
        a0_pois = np.array([[poismean, 0, 0, 0, 0, 0, 0, 0]])
        R0_pois = np.diag([.1, .1, .5, .5, .5, .5, .5, .5])
        mod = dcmm(a0_bern = a0_bern, R0_bern = R0_bern,
                    nregn_bern = 1,
                    ntrend_bern = 1,
                    seasPeriods_bern = seasPeriods,
                    seasHarmComponents_bern = seasHarmComponents,
                    deltrend_bern = .99, delregn_bern = .995,
                    delhol_bern = 1, delseas_bern = .995,
              a0_pois = a0_pois, R0_pois = R0_pois,
                    nregn_pois = 1,
                    ntrend_pois = 1,
                    seasPeriods_pois = seasPeriods,
                    seasHarmComponents_pois = seasHarmComponents,
                    deltrend_pois = .99, delregn_pois = .995,
                    delhol_pois = 1, delseas_pois = .995)
    elif multiscale:
        # Define a multiscale DCMM for that single item's sales
        a0_bern = np.array([[bernmean, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
        R0_bern = np.identity(9)/2
        a0_pois = np.array([[poismean, 0, 0, 0, 0, 0, 0, 0, 0]])
        R0_pois = np.diag([.1, .1, .5, .5, .5, .5, .5, .5, .5])
        mod = dcmm(a0_bern = a0_bern, R0_bern = R0_bern,
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
        
    return mod

def define_dbcm(Y_transaction, X_transaction = None, Y_cascade = None, X_cascade = None, prior_length = 30, seasPeriods = [7], seasHarmComponents = [[1,2,3]], multiscale = False):
    def ncol(x):
        if len(np.shape(x)) == 1:
            return 1
        else:
            return np.shape(x)[1]
        
    Y = Y_transaction
    bernmean = len(Y[:prior_length].nonzero()[0])/prior_length
    poismean = Y[:prior_length].mean()
    pbern = ppois = 1 + ncol(X_cascade)
    
    def cascade_prior_mean(alpha, beta):
        alpha += 1
        beta += 1
        mean = alpha / (alpha + beta)
        logit_mean = np.log(mean / (1 - mean))
        return logit_mean
    
    Yc = np.c_[Y_transaction - Y_cascade[:,0], Y_cascade]
    Yc = np.sum(Yc[:prior_length], axis = 0)
    ncascade = 4
    pcascade = 1 + ncol(X_cascade)
    means = [cascade_prior_mean(Yc[i+1], Yc[i]) for i in range(ncascade)]
    a0_cascade = [np.array([m, 0]).reshape(-1,1) for m in means]
    R0_cascade = [np.array([[.1, 0], [0, 0.1]]) for i in range(ncascade)]
    
    if not multiscale:
        a0_bern = np.array([[bernmean, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
        R0_bern = np.identity(8)/2
        a0_pois = np.array([[poismean, 0, 0, 0, 0, 0, 0, 0]])
        R0_pois = np.diag([.1, .1, .5, .5, .5, .5, .5, .5])
        mod = dbcm(a0_bern = a0_bern, R0_bern = R0_bern,
                    nregn_bern = 1,
                    ntrend_bern = 1,
                    seasPeriods_bern = seasPeriods,
                    seasHarmComponents_bern = seasHarmComponents,
                    deltrend_bern = .99, delregn_bern = .995,
                    delhol_bern = 1, delseas_bern = .995,
                   
              a0_pois = a0_pois, R0_pois = R0_pois,
                    nregn_pois = 1,
                    ntrend_pois = 1,
                    seasPeriods_pois = seasPeriods,
                    seasHarmComponents_pois = seasHarmComponents,
                    deltrend_pois = .99, delregn_pois = .995,
                    delhol_pois = 1, delseas_pois = .995,
              
              ncascade = ncascade,
                 a0_cascade = a0_cascade, # List of length ncascade
                 R0_cascade = R0_cascade, # List of length ncascade
                 nregn_cascade = 1,
                 ntrend_cascade = 1,
                 nmultiscale_cascade = 0,
                 seasPeriods_cascade = None,
                 seasHarmComponents_cascade = None,
                 deltrend_cascade = .99, delregn_cascade = .995,
                 delhol_cascade = 1, delseas_cascade = 1,
                 delmultiscale_cascade = 1)
        
    elif multiscale:
        a0_bern = np.array([[bernmean, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
        R0_bern = np.identity(9)/2
        a0_pois = np.array([[poismean, 0, 0, 0, 0, 0, 0, 0, 0]])
        R0_pois = np.diag([.1, .1, .5, .5, .5, .5, .5, .5, .5])
        mod = dbcm(a0_bern = a0_bern, R0_bern = R0_bern,
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
                    delmultiscale_pois = .995,
                  
               ncascade = ncascade,
                 a0_cascade = a0_cascade, # List of length ncascade
                 R0_cascade = R0_cascade, # List of length ncascade
                 nregn_cascade = 1,
                 ntrend_cascade = 1,
                 nmultiscale_cascade = 0, # No multiscale cascades, only the DCMM portion is multiscale
                 seasPeriods_cascade = None,
                 seasHarmComponents_cascade = None,
                 deltrend_cascade = .99, delregn_cascade = .995,
                 delhol_cascade = 1, delseas_cascade = 1,
                 delmultiscale_cascade = 1)
        
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