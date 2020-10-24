import numpy as np
from .latent_factor_fxns import forecast_marginal_lf_dcmm, forecast_path_lf_dcmm
from .dglm import bern_dglm, pois_dglm
from .update import update_F

class dcmm:
    def __init__(self,
                 a0_bern = None,
                 R0_bern = None,
                 nregn_bern = 0,
                 ntrend_bern = 0,
                 nlf_bern = 0,
                 nhol_bern = 0,
                 seasPeriods_bern = [],
                 seasHarmComponents_bern = [],
                 deltrend_bern = 1, delregn_bern = 1,
                 delhol_bern = 1,
                 delseas_bern = 1, dellf_bern = 1,
                 a0_pois = None,
                 R0_pois = None,
                 nregn_pois = 0,
                 ntrend_pois = 0,
                 nlf_pois = 0,
                 nhol_pois = 0,
                 seasPeriods_pois = [],
                 seasHarmComponents_pois = [],
                 deltrend_pois = 1, delregn_pois = 1,
                 delhol_pois = 1,
                 delseas_pois = 1, dellf_pois = 1,
                 rho = 1,
                 interpolate=True,
                 adapt_discount=False):
        """
        :param a0_bern: Prior mean vector for bernoulli DGLM
        :param R0_bern: Prior covariance matrix for bernoulli DGLM
        :param nregn_bern: Number of regression components in bernoulli DGLM
        :param ntrend_bern: Number of trend components in bernoulli DGLM
        :param nlf_bern: Number of latent factor components in bernoulli DGLM
        :param seasPeriods_bern: List of periods of seasonal components in bernoulli DGLM
        :param seasHarmComponents_bern: List of harmonic components included for each period in bernoulli DGLM
        :param deltrend_bern: Discount factor on trend components in bernoulli DGLM
        :param delregn_bern: Discount factor on regression components in bernoulli DGLM
        :param delhol_bern: Discount factor on holiday component in bernoulli DGLM (currently deprecated)
        :param delseas_bern: Discount factor on seasonal components in bernoulli DGLM
        :param dellf_bern: Discount factor on latent factor components in bernoulli DGLM
        :param a0_pois: Prior mean vector for poisson DGLM
        :param R0_pois: Prior covariance matrix for poisson DGLM
        :param nregn_pois: Number of regression components in poisson DGLM
        :param ntrend_pois: Number of trend components in poisson DGLM
        :param nlf_pois: Number of latent factor components in poisson DGLM
        :param seasPeriods_pois: List of periods of seasonal components in poisson DGLM
        :param seasHarmComponents_pois: List of harmonic components included for each period in poisson DGLM
        :param deltrend_pois: Discount factor on trend components in poisson DGLM
        :param delregn_pois: Discount factor on regression components in poisson DGLM
        :param delhol_pois: Discount factor on holiday component in poisson DGLM (currently deprecated)
        :param delseas_pois: Discount factor on seasonal components in poisson DGLM
        :param dellf_pois: Discount factor on latent factor components in poisson DGLM
        :param rho: Discount factor for random effects extension in poisson DGLM (smaller rho increases variance)
        """

        self.bern_mod = bern_dglm(a0=a0_bern,
                                  R0=R0_bern,
                                  nregn=nregn_bern,
                                  ntrend=ntrend_bern,
                                  nlf=nlf_bern,
                                  nhol=nhol_bern,
                                  seasPeriods=seasPeriods_bern,
                                  seasHarmComponents=seasHarmComponents_bern,
                                  deltrend=deltrend_bern, delregn=delregn_bern,
                                  delhol=delhol_bern, delseas=delseas_bern,
                                  dellf=dellf_bern,
                                  interpolate=interpolate,
                                  adapt_discount=adapt_discount)

        self.pois_mod = pois_dglm(a0=a0_pois,
                                  R0=R0_pois,
                                  nregn=nregn_pois,
                                  ntrend=ntrend_pois,
                                  nlf=nlf_pois,
                                  nhol=nhol_pois,
                                  seasPeriods=seasPeriods_pois,
                                  seasHarmComponents=seasHarmComponents_pois,
                                  deltrend=deltrend_pois, delregn=delregn_pois,
                                  delhol=delhol_pois, delseas=delseas_pois,
                                  dellf=dellf_pois,
                                  rho=rho,
                                  interpolate=interpolate,
                                  adapt_discount=adapt_discount)

        self.t = 0

        
    # X is a list or tuple of length 2. The first component is data for the bernoulli DGLM, the next is for the Poisson DGLM.
    def update(self, y = None, X = None):
        X = self.make_pair(X)

        if y is None:
            self.bern_mod.update(y=y)
            self.pois_mod.update(y=y)
        elif y == 0:
            self.bern_mod.update(y = 0, X = X[0])
            self.pois_mod.update(y = np.nan, X = X[1])
        else: # only update beta model if we have significant uncertainty in the forecast
            # get the lower end forecast on the logit scale
            F = update_F(self.bern_mod, X[0], F=self.bern_mod.F.copy())
            ft, qt = self.bern_mod.get_mean_and_var(F, self.bern_mod.a, self.bern_mod.R)
            fcast_logit_lb = ft - np.sqrt(qt)
            # translate to a prod for a rough idea of whether we're already pretty confident for this forecast
            if 1 / (1 + np.exp(-fcast_logit_lb)) < 0.975:
                self.bern_mod.update(y=1, X = X[0])
            else:
                self.bern_mod.update(y=np.nan, X=X[0])
            self.pois_mod.update(y = y - 1, X = X[1]) # Shifted Y values in the Poisson DGLM
            
        self.t += 1
        
    def update_lf_sample(self, y = None, X = None, phi_samps = None, parallel=False):
        X = self.make_pair(X)
        phi_samps = self.make_pair(phi_samps)

        if y is None:
            self.bern_mod.update_lf_sample(y=y)
            self.pois_mod.update_lf_sample(y=y)
        elif y == 0:
            self.bern_mod.update_lf_sample(y = 0, X = X[0], phi_samps = phi_samps[0], parallel = parallel)
            self.pois_mod.update_lf_sample(y = np.nan, X = X[1], phi_samps = phi_samps[1], parallel = parallel)
        else:
            self.bern_mod.update_lf_sample(y = 1, X = X[0], phi_samps = phi_samps[0], parallel = parallel)
            # Shifted Y values in the Poisson DGLM
            self.pois_mod.update_lf_sample(y =y - 1, X = X[1], phi_samps = phi_samps[1], parallel = parallel)
            
        self.t += 1
        
    def update_lf_analytic(self, y = None, X = None, phi_mu = None, phi_sigma = None):
        X = self.make_pair(X)
        phi_mu = self.make_pair(phi_mu)
        phi_sigma = self.make_pair(phi_sigma)

        if y is None:
            self.bern_mod.update_lf_analytic(y=y)
            self.pois_mod.update_lf_analytic(y=y)
        elif y == 0:
            self.bern_mod.update_lf_analytic(y = 0, X = X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0])
            self.pois_mod.update_lf_analytic(y = np.nan, X = X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1])
        else:
            self.bern_mod.update_lf_analytic(y = 1, X = X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0])
            # Shifted Y values in the Poisson DGLM
            self.pois_mod.update_lf_analytic(y =y - 1, X = X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1])
            
        self.t += 1
            
    def forecast_marginal(self, k, X = None, nsamps = 1, mean_only = False, state_mean_var = False):
        X = self.make_pair(X)

        if mean_only:
            mean_bern = self.bern_mod.forecast_marginal(k, X[0], nsamps, mean_only)
            mean_pois = self.pois_mod.forecast_marginal(k, X[1], nsamps, mean_only)
            return mean_bern * (mean_pois + 1)
        elif state_mean_var:
            mv_bern = self.bern_mod.forecast_marginal(k, X[0], state_mean_var = state_mean_var)
            mv_pois = self.pois_mod.forecast_marginal(k, X[1], state_mean_var = state_mean_var)
            return mv_bern, mv_pois
        else:
            samps_bern = self.bern_mod.forecast_marginal(k, X[0], nsamps)
            samps_pois = self.pois_mod.forecast_marginal(k, X[1], nsamps) + np.ones([nsamps]) # Shifted Y values in the Poisson DGLM
            return samps_bern * samps_pois
    
    def forecast_marginal_lf_analytic(self, k, X = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False, state_mean_var = False):
        X = self.make_pair(X)
        phi_mu = self.make_pair(phi_mu)
        phi_sigma = self.make_pair(phi_sigma)

        if mean_only:
            mean_bern = self.bern_mod.forecast_marginal_lf_analytic(k, X[0], phi_mu[0], phi_sigma[0], nsamps, mean_only)
            mean_pois = self.pois_mod.forecast_marginal_lf_analytic(k, X[1], phi_mu[1], phi_sigma[1], nsamps, mean_only)
            return np.array([[mean_bern * (mean_pois + 1)]])
        elif state_mean_var:
            mv_bern = self.bern_mod.forecast_marginal_lf_analytic(k, X[0], phi_mu[0], phi_sigma[0], state_mean_var = state_mean_var)
            mv_pois = self.pois_mod.forecast_marginal_lf_analytic(k, X[1], phi_mu[1], phi_sigma[1], state_mean_var = state_mean_var)
            return mv_bern, mv_pois
        else:
            samps_bern = self.bern_mod.forecast_marginal_lf_analytic(k, X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0], nsamps = nsamps)
            samps_pois = self.pois_mod.forecast_marginal_lf_analytic(k, X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1], nsamps = nsamps) + np.ones([nsamps]) # Shifted Y values in the Poisson DGLM
            return samps_bern * samps_pois

    def forecast_marginal_lf_analytic_new(self, k, X = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False, state_mean_var = False):
        X = self.make_pair(X)
        phi_mu = self.make_pair(phi_mu)
        phi_sigma = self.make_pair(phi_sigma)

        if mean_only:
            mean_bern = self.bern_mod.forecast_marginal_lf_analytic(k, X[0], phi_mu[0], phi_sigma[0], nsamps, mean_only)
            mean_pois = self.pois_mod.forecast_marginal_lf_analytic(k, X[1], phi_mu[1], phi_sigma[1], nsamps, mean_only)
            return np.array([[mean_bern * (mean_pois + 1)]])
        elif state_mean_var:
            mv_bern = self.bern_mod.forecast_marginal_lf_analytic(k, X[0], phi_mu[0], phi_sigma[0], state_mean_var = state_mean_var)
            mv_pois = self.pois_mod.forecast_marginal_lf_analytic(k, X[1], phi_mu[1], phi_sigma[1], state_mean_var = state_mean_var)
            return mv_bern, mv_pois
        else:
            return forecast_marginal_lf_dcmm(self, k, X[0], phi_mu[0], phi_sigma[0], nsamps=nsamps)

    def forecast_marginal_lf_sample(self, k, X = None, phi_samps = None, nsamps = 1, mean_only = False):
        X = self.make_pair(X)
        phi_samps = self.make_pair(phi_samps)

        samps_bern = self.bern_mod.forecast_marginal_lf_sample(k, X[0], phi_samps[0], mean_only)
        samps_pois = self.pois_mod.forecast_marginal_lf_sample(k, X[1], phi_samps[1], mean_only) + np.ones([nsamps]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois

    def forecast_path_lf_sample(self, k, X = None, phi_samps=None, nsamps = 1):
        X = self.make_pair(X)
        phi_samps = self.make_pair(phi_samps)

        samps_bern = self.bern_mod.forecast_path_lf_sample(k, X[0], phi_samps[0], nsamps)
        samps_pois = self.pois_mod.forecast_path_lf_sample(k, X[1], phi_samps[1], nsamps) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
    
    def forecast_path(self, k, X = None, nsamps = 1):
        X = self.make_pair(X)

        samps_bern = self.bern_mod.forecast_path(k, X[0], nsamps)
        samps_pois = self.pois_mod.forecast_path(k, X[1], nsamps) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
    
    def forecast_path_copula(self, k, X = None, nsamps = 1, **kwargs):
        X = self.make_pair(X)

        samps_bern = self.bern_mod.forecast_path_copula(k, X[0], nsamps, **kwargs)
        samps_pois = self.pois_mod.forecast_path_copula(k, X[1], nsamps, **kwargs) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois
    
    def forecast_path_lf_copula(self, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = None, nsamps = 1, **kwargs):
        X = self.make_pair(X)
        if k == 2 and isinstance(phi_mu, (list, tuple)):
            if not isinstance(phi_mu[0], (list, tuple)):
                phi_mu = (phi_mu, phi_mu)
                phi_sigma = (phi_sigma, phi_sigma)
                phi_psi = (phi_psi, phi_psi)
        else:
            phi_mu = self.make_pair(phi_mu)
            phi_sigma = self.make_pair(phi_sigma)
            phi_psi = self.make_pair(phi_psi)

        samps_bern = self.bern_mod.forecast_path_lf_copula(k, X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0], phi_psi = phi_psi[0], nsamps = nsamps, **kwargs)
        samps_pois = self.pois_mod.forecast_path_lf_copula(k, X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1], phi_psi = phi_psi[1], nsamps = nsamps, **kwargs) + np.ones([nsamps, k]) # Shifted Y values in the Poisson DGLM
        return samps_bern * samps_pois

    def forecast_path_lf_copula_new(self, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = None, nsamps = 1, **kwargs):
        X = self.make_pair(X)
        if k == 2 and isinstance(phi_mu, (list, tuple)):
            if not isinstance(phi_mu[0], (list, tuple)):
                phi_mu = (phi_mu, phi_mu)
                phi_sigma = (phi_sigma, phi_sigma)
                phi_psi = (phi_psi, phi_psi)
        else:
            phi_mu = self.make_pair(phi_mu)
            phi_sigma = self.make_pair(phi_sigma)
            phi_psi = self.make_pair(phi_psi)

        return forecast_path_lf_dcmm(self, k, X[0], phi_mu[0], phi_sigma[0], phi_psi[0], nsamps=nsamps, **kwargs)


    def forecast_path_lf_copula_density(self, y, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = (None, None), nsamps = 1, **kwargs):
        X = self.make_pair(X)
        phi_mu = self.make_pair(phi_mu)
        phi_sigma = self.make_pair(phi_sigma)
        phi_psi = self.make_pair(phi_psi)

        z = np.zeros([k])
        y = y.reshape(-1)
        z[y > 0] = 1
        logdens_bern = self.bern_mod.forecast_path_lf_copula(k, X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0], phi_psi = phi_psi[0], nsamps = nsamps, y = z, **kwargs)
        # Shifted Y values in the Poisson DGLM
        y = y - 1
        y = y.astype('float')
        # 0's in the original data (now -1's) are considered 'missing by the Poisson model
        y[y < 0] = np.nan
        logdens_pois = self.pois_mod.forecast_path_lf_copula(k, X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1], phi_psi = phi_psi[1], nsamps = nsamps, y = y, **kwargs)
        return logdens_bern, logdens_pois

    def forecast_state_mean_and_var(self, k = 1, X = None):
        mean_var_bern = self.bern_mod.forecast_state_mean_and_var(k, X[0])
        mean_var_pois = self.pois_mod.forecast_state_mean_and_var(k, X[1])
        return mean_var_bern, mean_var_pois

    def make_pair(self, x):
        if isinstance(x, (list, tuple)):
            if len(x) == 2:
                return x
            else:
                return (x, x)
        else:
            return (x, x)