import numpy as np
import scipy as sc
from forecasting.dglm import *
from forecasting.dcmm import *
from forecasting.update import *
from forecasting.forecast import *
from forecasting.seasonal import *

class dbcm:
    def __init__(self,
                 a0_bern = None, 
                 R0_bern = None,
                 nregn_bern = None,
                 ntrend_bern = 0,
                 nmultiscale_bern = 0,
                 seasPeriods_bern = None,
                 seasHarmComponents_bern = None,
                 deltrend_bern = 1, delregn_bern = 1,
                 delhol_bern = 1, delseas_bern = 1,
                 delmultiscale_bern = 1,
                 
                 a0_pois = None, 
                 R0_pois = None,
                 nregn_pois = None,
                 ntrend_pois = 0,
                 nmultiscale_pois = 0,
                 seasPeriods_pois = None,
                 seasHarmComponents_pois = None,
                 deltrend_pois = 1, delregn_pois = 1,
                 delhol_pois = 1, delseas_pois = 1,
                 delmultiscale_pois = 1,
                
                 ncascade = 4,
                 a0_cascade = None, # List of length ncascade
                 R0_cascade = None, # List of length ncascade
                 nregn_cascade = None,
                 ntrend_cascade = 0,
                 nmultiscale_cascade = 0,
                 seasPeriods_cascade = None,
                 seasHarmComponents_cascade = None,
                 deltrend_cascade = 1, delregn_cascade = 1,
                 delhol_cascade = 1, delseas_cascade = 1,
                 delmultiscale_cascade = 1):
        
        self.dcmm = dcmm(a0_bern, 
                 R0_bern,
                 nregn_bern,
                 ntrend_bern,
                 nmultiscale_bern,
                 seasPeriods_bern,
                 seasHarmComponents_bern,
                 deltrend_bern, delregn_bern,
                 delhol_bern, delseas_bern,
                 delmultiscale_bern,
                         
                 a0_pois, 
                 R0_pois,
                 nregn_pois,
                 ntrend_pois,
                 nmultiscale_pois,
                 seasPeriods_pois,
                 seasHarmComponents_pois,
                 deltrend_pois, delregn_pois,
                 delhol_pois, delseas_pois,
                 delmultiscale_pois)
        
        self.ncascade = ncascade
        self.cascade = list(map(lambda a0, R0: bin_dglm(a0, R0, nregn_cascade, ntrend_cascade,
                                                       nmultiscale_cascade, seasPeriods_cascade,
                                                       seasHarmComponents_cascade, deltrend_cascade, delregn_cascade,
                                                       delhol_cascade, delseas_cascade, delmultiscale_cascade),
                                a0_cascade, R0_cascade))
        
        self.t = 0
        
    # X is a list or tuple of length 3.
    # Data for the bernoulli DGLM, the Poisson DGLM, and then the cascade
    # Note we assume that all binomials in the cascade have the same regression components
    def update(self, y_transaction = None, X_transaction = None, y_cascade = None, X_cascade = None):
        # Update the DCMM for transactions
        # We assume the bernoulli and poisson DGLMs have the same regression components
        self.dcmm.update(y_transaction, (X_transaction, X_transaction))
        
        # Update the cascade of binomial DGLMs for basket sizes
        # Note that we pass in n, p as a tuple
        update(self.cascade[0], (y_transaction, y_cascade[0]), X_cascade)
        for i in range(1, self.ncascade):
            update(self.cascade[i], (y_cascade[i-1], y_cascade[i]), X_cascade)
            
        self.t += 1
        
    # Note we assume that the cascade is NOT multiscale, only the DCMM for transactions 
    def multiscale_update(self, y_transaction = None, X_transaction = None, y_cascade = None, X_cascade = None, phi_samps = None):
        self.dcmm.multiscale_update(y_transaction, (X_transaction, X_transaction), (phi_samps, phi_samps))
        
        # Update the cascade of binomial DGLMs for basket sizes
        # Note that we pass in n, p as a tuple
        update(self.cascade[0], (y_transaction, y_cascade[0]), X_cascade)
        for i in range(1, self.ncascade):
            update(self.cascade[i], (y_cascade[i-1], y_cascade[i]), X_cascade)
            
        self.t += 1
        
    def multiscale_update_approx(self, y_transaction = None, X_transaction = None, y_cascade = None, X_cascade = None, phi_mu = None, phi_sigma = None):
        self.dcmm.multiscale_update_approx(y_transaction,
                                           (X_transaction, X_transaction),
                                           (phi_mu, phi_mu),
                                           (phi_sigma, phi_sigma))
        
        # Update the cascade of binomial DGLMs for basket sizes
        # Note that we pass in n, p as a tuple
        update(self.cascade[0], (y_transaction, y_cascade[0]), X_cascade)
        for i in range(1, self.ncascade):
            update(self.cascade[i], (y_cascade[i-1], y_cascade[i]), X_cascade)
                
        self.t += 1
            
    def forecast_marginal(self, k, X_transaction = None, X_cascade = None, nsamps = 1, mean_only = False):
        sales_samps = np.zeros(nsamps)
        transaction_samps = self.dcmm.forecast_marginal(k, X_transaction, nsamps, mean_only)
        cascade_samps[0] = forecast_marginal(self.cascade[0], k, X_cascade, nsamps, mean_only, n = transaction_samps)
        for i in range(1, self.ncascade):
            cascade_samps.append(self.cascade[i].simulate())

    
    def multiscale_forecast_marginal_approx(self, k, X = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False):
        if mean_only:
            mean_bern = multiscale_forecast_marginal_approx(self.bern_mod, k, X[0], phi_mu[0], phi_sigma[0], nsamps, mean_only)
            mean_pois = multiscale_forecast_marginal_approx(self.pois_mod, k, X[1], phi_mu[1], phi_sigma[1], nsamps, mean_only)
            return mean_bern * (mean_pois + 1)
        
        else:
            samps_bern = multiscale_forecast_marginal_approx(self.bern_mod, k, X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0], nsamps = nsamps)
            samps_pois = multiscale_forecast_marginal_approx(self.pois_mod, k, X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1], nsamps = nsamps) + np.ones([nsamps]) # Shifted Y values in the Poisson DGLM
            return samps_bern * samps_pois
        
    def multiscale_forecast_marginal(self, k, X = None, phi_samps = None, mean_only = False):
        nsamps = phi_samps[1].shape[0]
        samps_bern = multiscale_forecast_marginal(self.bern_mod, k, X[0], phi_samps = phi_samps[0])
        samps_pois = multiscale_forecast_marginal(self.pois_mod, k, X[1], phi_samps = phi_samps[1]) + np.ones([nsamps]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
    
    def forecast_path(self, k, X = None, nsamps = 1):
        samps_bern = forecast_path(self.bern_mod, k, X[0], nsamps)
        samps_pois = forecast_path(self.pois_mod, k, X[1], nsamps) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
    
    def forecast_path_approx(self, k, X = None, nsamps = 1):
        samps_bern = forecast_path_approx(self.bern_mod, k, X[0], nsamps)
        samps_pois = forecast_path_approx(self.pois_mod, k, X[1], nsamps) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
    
    def multiscale_forecast_path_approx(self, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = None, nsamps = 1):
        samps_bern = multiscale_forecast_path_approx(self.bern_mod, k, X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0], phi_psi = phi_psi[0], nsamps = nsamps)
        samps_pois = multiscale_forecast_path_approx(self.pois_mod, k, X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1], phi_psi = phi_psi[0], nsamps = nsamps) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
        
        