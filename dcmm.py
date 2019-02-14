import numpy as np
import scipy as sc
from forecasting.dglm import *
from forecasting.update import *
from forecasting.forecast import *
from forecasting.seasonal import *

class dcmm:
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
                 delmultiscale_pois = 1):
        
        self.bern_mod = bern_dglm(a0 = a0_bern,
                            R0 = R0_bern,
                            nregn = nregn_bern,
                            ntrend = ntrend_bern,
                            nmultiscale = nmultiscale_bern,
                            seasPeriods = seasPeriods_bern,
                            seasHarmComponents = seasHarmComponents_bern,
                            deltrend = deltrend_bern, delregn = delregn_bern,
                            delhol = delhol_bern, delseas = delseas_bern,
                            delmultiscale = delmultiscale_bern)
        
        self.pois_mod = pois_dglm(a0 = a0_pois,
                            R0 = R0_pois,
                            nregn = nregn_pois,
                            ntrend = ntrend_pois,
                            nmultiscale = nmultiscale_pois,
                            seasPeriods = seasPeriods_pois,
                            seasHarmComponents = seasHarmComponents_pois,
                            deltrend = deltrend_pois, delregn = delregn_pois,
                            delhol = delhol_pois, delseas = delseas_pois,
                            delmultiscale = delmultiscale_pois)
        
        self.t = 0
        
    # X is a list or tuple of length 2. The first component is data for the bernoulli DGLM, the next is for the Poisson DGLM.
    def update(self, y = None, X = None):
        if y == 0:
            update(self.bern_mod, y = 0, X = X[0])
            update(self.pois_mod, y = np.nan, X = X[1])
        else:
            update(self.bern_mod, y = 1, X = X[0])
            update(self.pois_mod, y = y - 1, X = X[1]) # Shifted Y values in the Poisson DGLM
            
        self.t += 1
        
    def multiscale_update(self, y = None, X = None, phi_samps = None):
        if y == 0:
            multiscale_update(self.bern_mod, y = 0, X = X[0], phi_samps = phi_samps[0])
            multiscale_update(self.pois_mod, y = np.nan, X = X[1], phi_samps = phi_samps[1])
        else:
            multiscale_update(self.bern_mod, y = 1, X = X[0], phi_samps = phi_samps[0])
            # Shifted Y values in the Poisson DGLM
            multiscale_update(self.pois_mod, y = y - 1, X = X[1], phi_samps = phi_samps[1]) 
            
        self.t += 1
        
    def multiscale_update_approx(self, y = None, X = None, phi_mu = None, phi_sigma = None):
        if y == 0:
            multiscale_update_approx(self.bern_mod, y = 0, X = X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0])
            multiscale_update_approx(self.pois_mod, y = np.nan, X = X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1])
        else:
            multiscale_update_approx(self.bern_mod, y = 1, X = X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0])
            # Shifted Y values in the Poisson DGLM
            multiscale_update_approx(self.pois_mod, y = y - 1, X = X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1]) 
            
        self.t += 1
            
    def forecast_marginal(self, k, X = None, nsamps = 1, mean_only = False): 
        if mean_only:
            mean_bern = forecast_marginal(self.bern_mod, k, X[0], nsamps, mean_only)
            mean_pois = forecast_marginal(self.pois_mod, k, X[1], nsamps, mean_only)
            return mean_bern * (mean_pois + 1)
        else:
            samps_bern = forecast_marginal(self.bern_mod, k, X[0], nsamps)
            samps_pois = forecast_marginal(self.pois_mod, k, X[1], nsamps) + np.ones([nsamps]) # Shifted Y values in the Poisson DGLM
            return samps_bern * samps_pois
    
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
        
        