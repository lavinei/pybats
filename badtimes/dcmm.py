import numpy as np
import scipy as sc
from .dglm import *
from .update import *
from .forecast import *
from .seasonal import *

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
                 delmultiscale_pois = 1,
                 rho = 1):
        
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
                            delmultiscale = delmultiscale_pois,
                            rho = rho)
        
        self.t = 0
        
    # X is a list or tuple of length 2. The first component is data for the bernoulli DGLM, the next is for the Poisson DGLM.
    def update(self, y = None, X = None):
        if y == 0:
            self.bern_mod.update(y = 0, X = X[0])
            self.pois_mod.update(y = np.nan, X = X[1])
        else:
            self.bern_mod.update(y = 1, X = X[0])
            self.pois_mod.update(y = y - 1, X = X[1]) # Shifted Y values in the Poisson DGLM
            
        self.t += 1
        
    def multiscale_update(self, y = None, X = None, phi_samps = None, parallel=False):
        if y == 0:
            self.bern_mod.multiscale_update(y = 0, X = X[0], phi_samps = phi_samps[0], parallel = parallel)
            self.pois_mod.multiscale_update(y = np.nan, X = X[1], phi_samps = phi_samps[1], parallel = parallel)
        else:
            self.bern_mod.multiscale_update(y = 1, X = X[0], phi_samps = phi_samps[0], parallel = parallel)
            # Shifted Y values in the Poisson DGLM
            self.pois_mod.multiscale_update(y = y - 1, X = X[1], phi_samps = phi_samps[1], parallel = parallel)
            
        self.t += 1
        
    def multiscale_update_approx(self, y = None, X = None, phi_mu = None, phi_sigma = None):
        if y == 0:
            self.bern_mod.multiscale_update_approx(y = 0, X = X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0])
            self.pois_mod.multiscale_update_approx(y = np.nan, X = X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1])
        else:
            self.bern_mod.multiscale_update_approx(y = 1, X = X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0])
            # Shifted Y values in the Poisson DGLM
            self.pois_mod.multiscale_update_approx(y = y - 1, X = X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1])
            
        self.t += 1
            
    def forecast_marginal(self, k, X = None, nsamps = 1, mean_only = False):
        if mean_only:
            mean_bern = self.bern_mod.forecast_marginal(k, X[0], nsamps, mean_only)
            mean_pois = self.pois_mod.forecast_marginal(k, X[1], nsamps, mean_only)
            return mean_bern * (mean_pois + 1)
        else:
            samps_bern = self.bern_mod.forecast_marginal(k, X[0], nsamps)
            samps_pois = self.pois_mod.forecast_marginal(k, X[1], nsamps) + np.ones([nsamps]) # Shifted Y values in the Poisson DGLM
            return samps_bern * samps_pois
    
    def multiscale_forecast_marginal_approx(self, k, X = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False):
        if mean_only:
            mean_bern = self.bern_mod.multiscale_forecast_marginal_approx(k, X[0], phi_mu[0], phi_sigma[0], nsamps, mean_only)
            mean_pois = self.pois_mod.multiscale_forecast_marginal_approx(k, X[1], phi_mu[1], phi_sigma[1], nsamps, mean_only)
            return np.array([[mean_bern * (mean_pois + 1)]])
        
        else:
            samps_bern = self.bern_mod.multiscale_forecast_marginal_approx(k, X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0], nsamps = nsamps)
            samps_pois = self.pois_mod.multiscale_forecast_marginal_approx(k, X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1], nsamps = nsamps) + np.ones([nsamps]) # Shifted Y values in the Poisson DGLM
            return samps_bern * samps_pois
        
    def multiscale_forecast_marginal(self, k, X = None, phi_samps = None, nsamps = 1, mean_only = False):
        samps_bern = self.bern_mod.multiscale_forecast_marginal(k, X[0], phi_samps[0], mean_only)
        samps_pois = self.pois_mod.multiscale_forecast_marginal(k, X[1], phi_samps[1], mean_only) + np.ones([nsamps]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
    
    def forecast_path(self, k, X = None, nsamps = 1):
        samps_bern = self.bern_mod.forecast_path(k, X[0], nsamps)
        samps_pois = self.pois_mod.forecast_path(k, X[1], nsamps) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
    
    def forecast_path_approx(self, k, X = None, nsamps = 1):
        samps_bern = self.bern_mod.forecast_path_approx(k, X[0], nsamps)
        samps_pois = self.pois_mod.forecast_path_approx(k, X[1], nsamps) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
    
    def multiscale_forecast_path_approx(self, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = (None, None), nsamps = 1):
        samps_bern = self.bern_mod.multiscale_forecast_path_approx(k, X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0], phi_psi = phi_psi[0], nsamps = nsamps)
        samps_pois = self.pois_mod.multiscale_forecast_path_approx(k, X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1], phi_psi = phi_psi[1], nsamps = nsamps) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
        
        