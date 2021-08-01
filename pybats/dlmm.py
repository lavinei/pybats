# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_dlmm.ipynb (unless otherwise specified).

__all__ = ['dlmm']

# Internal Cell
#exporti
import numpy as np
from .dglm import bern_dglm, dlm
from .update import update_F
from scipy.special import expit

# Cell
class dlmm:
    def __init__(self,
                 a0_bern = None,
                 R0_bern = None,
                 nregn_bern = 0,
                 ntrend_bern = 0,
                 nhol_bern = 0,
                 nlf_bern = 0,
                 seasPeriods_bern = [],
                 seasHarmComponents_bern = [],
                 deltrend_bern = 1, delregn_bern = 1,
                 delhol_bern = 1,
                 delseas_bern = 1, dellf_bern = 1,
                 rho = 1,
                 a0_dlm = None,
                 R0_dlm = None,
                 n0_dlm = 1,
                 s0_dlm = 1,
                 nregn_dlm = 0,
                 ntrend_dlm = 0,
                 nhol_dlm = 0,
                 nlf_dlm = 0,
                 seasPeriods_dlm = [],
                 seasHarmComponents_dlm = [],
                 deltrend_dlm = 1, delregn_dlm = 1,
                 delhol_dlm = 1,
                 delseas_dlm = 1,
                 delVar_dlm = 1, dellf_dlm = 1,
                 interpolate=True,
                 adapt_discount=False):
        """
        :param a0_bern: Prior mean vector for bernoulli DGLM
        :param R0_bern: Prior covariance matrix for bernoulli DGLM
        :param nregn_bern: Number of regression components in bernoulli DGLM
        :param ntrend_bern: Number of trend components in bernoulli DGLM
        :param seasPeriods_bern: List of periods of seasonal components in bernoulli DGLM
        :param seasHarmComponents_bern: List of harmonic components included for each period in bernoulli DGLM
        :param deltrend_bern: Discount factor on trend components in bernoulli DGLM
        :param delregn_bern: Discount factor on regression components in bernoulli DGLM
        :param delhol_bern: Discount factor on holiday component in bernoulli DGLM (currently deprecated)
        :param delseas_bern: Discount factor on seasonal components in bernoulli DGLM
        :param rho: random effect discount factor for bernoulli DGLM (smaller rho increases variance)
        :param a0_dlm: Prior mean vector for normal DLM
        :param R0_dlm: Prior covariance matrix for normal DLM
        :param n0_dlm: Prior sample size for the dlm variance
        :param s0_dlm: Prior mean for the dlm variance
        :param nregn_dlm: Number of regression components in normal DLM
        :param ntrend_dlm: Number of trend components in normal DLM
        :param seasPeriods_dlm: List of periods of seasonal components in normal DLM
        :param seasHarmComponents_dlm: List of harmonic components included for each period in normal DLM
        :param deltrend_dlm: Discount factor on trend components in normal DLM
        :param delregn_dlm: Discount factor on regression components in normal DLM
        :param delhol_dlm: Discount factor on holiday component in normal DLM (currently deprecated)
        :param delseas_dlm: Discount factor on seasonal components in normal DLM
        :param delVar_dlm: Discount factor for observation volatility in normal DLM
        """

        self.bern_mod = bern_dglm(a0 = a0_bern,
                                  R0 = R0_bern,
                                  nregn = nregn_bern,
                                  ntrend = ntrend_bern,
                                  nlf = nlf_bern,
                                  nhol = nhol_bern,
                                  seasPeriods = seasPeriods_bern,
                                  seasHarmComponents = seasHarmComponents_bern,
                                  deltrend = deltrend_bern, delregn = delregn_bern,
                                  delhol = delhol_bern, delseas = delseas_bern,
                                  dellf = dellf_bern,
                                  rho = rho,
                                  interpolate = interpolate,
                                  adapt_discount = adapt_discount)

        self.dlm = dlm(a0 = a0_dlm,
                           R0 = R0_dlm,
                           n0 = n0_dlm,
                           s0 = s0_dlm,
                           nregn = nregn_dlm,
                           ntrend = ntrend_dlm,
                           nhol = nhol_dlm,
                           nlf = nlf_dlm,
                           seasPeriods = seasPeriods_dlm,
                           seasHarmComponents = seasHarmComponents_dlm,
                           deltrend = deltrend_dlm, delregn = delregn_dlm,
                           delhol = delhol_dlm, delseas = delseas_dlm,
                           dellf = dellf_dlm,
                           delVar = delVar_dlm,
                           interpolate = interpolate,
                           adapt_discount = adapt_discount)

        self.t = 0


    # X is a list or tuple of length 2. The first component is data for the bernoulli DGLM, the next is for the Normal DLM.
    def update(self, y = None, X = None):
        X = self.make_pair(X)
        if y is None:
            self.bern_mod.update(y=y)
            self.dlm.update(y=y)
        elif y == 0:
            self.bern_mod.update(y = 0, X = X[0])
            self.dlm.update(y = np.nan, X = X[1])
        else: # only update beta model if we have significant uncertainty in the forecast
            # get the lower end forecast on the logit scale
            F = update_F(self.bern_mod, X[0], F=self.bern_mod.F.copy())
            ft, qt = self.bern_mod.get_mean_and_var(F, self.bern_mod.a, self.bern_mod.R)
            fcast_logit_lb = ft - np.sqrt(qt)
            # translate to a prod for a rough idea of whether we're already pretty confident for this forecast
            if expit(fcast_logit_lb) < 0.975:
                self.bern_mod.update(y=1, X = X[0])
            else:
                self.bern_mod.update(y=np.nan, X=X[0])
            self.dlm.update(y = y, X = X[1]) # NO-Shifted Y values in the Normal DLM
        self.t += 1


    def update_lf_analytic(self, y = None, X = None, phi_mu = None, phi_sigma = None):

        X = self.make_pair(X)
        phi_mu = self.make_pair(phi_mu)
        phi_sigma = self.make_pair(phi_sigma)

        if y is None:
            self.bern_mod.update_lf_analytic(y = y)
            self.dlm.update_lf_analytic(y = y)
        elif y == 0:
            self.bern_mod.update_lf_analytic(y = 0, X = X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0])
            self.dlm.update_lf_analytic(y = np.nan, X = X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1])
        else:
            self.bern_mod.update_lf_analytic(y = 1, X = X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0])
            self.dlm.update_lf_analytic(y = y, X = X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1])

        self.t += 1

    def forecast_marginal(self, k, X = None, nsamps = 1, mean_only = False, state_mean_var = False):
        X = self.make_pair(X)

        if mean_only:
            mean_bern = self.bern_mod.forecast_marginal(k, X[0], nsamps, mean_only)
            mean_dlm = self.dlm.forecast_marginal(k, X[1], nsamps, mean_only)
            return mean_bern * (mean_dlm)
        elif state_mean_var:
            mv_bern = self.bern_mod.forecast_marginal(k, X[0], state_mean_var = state_mean_var)
            mv_dlm = self.dlm.forecast_marginal(k, X[1], state_mean_var = state_mean_var)
            return mv_bern, mv_dlm
        else:
            samps_bern = self.bern_mod.forecast_marginal(k, X[0], nsamps)
            samps_dlm = self.dlm.forecast_marginal(k, X[1], nsamps) # NO Shifted Y values in the normal DLM
            return samps_bern * samps_dlm


    def forecast_marginal_lf_analytic(self, k, X = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False, state_mean_var = False):
        X = self.make_pair(X)
        phi_mu = self.make_pair(phi_mu)
        phi_sigma = self.make_pair(phi_sigma)

        #print(X[0])

        if mean_only:
            mean_bern = self.bern_mod.forecast_marginal_lf_analytic(k, X[0], phi_mu[0], phi_sigma[0], nsamps, mean_only)
            mean_dlm = self.dlm.forecast_marginal_lf_analytic(k, X[1], phi_mu[1], phi_sigma[1], nsamps, mean_only)
            return np.array([[mean_bern * (mean_dlm)]])
        elif state_mean_var:
            mv_bern = self.bern_mod.forecast_marginal_lf_analytic(k, X[0], phi_mu[0], phi_sigma[0], state_mean_var = state_mean_var)
            mv_dlm = self.dlm.forecast_marginal_lf_analytic(k, X[1], phi_mu[1], phi_sigma[1], state_mean_var = state_mean_var)
            return mv_bern, mv_dlm
        else:
            samps_bern = self.bern_mod.forecast_marginal_lf_analytic(k, X[0], phi_mu = phi_mu[0], phi_sigma = phi_sigma[0], nsamps = nsamps)
            samps_dlm = self.dlm.forecast_marginal_lf_analytic(k, X[1], phi_mu = phi_mu[1], phi_sigma = phi_sigma[1], nsamps = nsamps)
            return samps_bern * samps_dlm


    def forecast_path(self, k, X = None, nsamps = 1):
        X = self.make_pair(X)

        samps_bern = self.bern_mod.forecast_path(k, X[0], nsamps)
        samps_dlm = self.dlm.forecast_path(k, X[1], nsamps) # NO Shifted Y values in the Normal DLM
        return samps_bern * samps_dlm

    def forecast_path_copula(self, k, X = None, nsamps = 1, **kwargs):
        X = self.make_pair(X)

        samps_bern = self.bern_mod.forecast_path_copula(k, X[0], nsamps, **kwargs)
        samps_dlm = self.dlm.forecast_path(k, X[1], nsamps) # NO Shifted Y values in the Normal DLM
        return samps_bern * samps_dlm

    def forecast_path_lf_copula(self, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = None, nsamps = 1, **kwargs):
        print('Path forecasting for latent factor DLMs is not yet implemented')

    def forecast_state_mean_and_var(self, k = 1, X = None):
        mean_var_bern = self.bern_mod.forecast_state_mean_and_var(k, X[0])
        mean_var_dlm = self.dlm.forecast_state_mean_and_var(k, X[1])
        return mean_var_bern, mean_var_dlm

    def make_pair(self, x):
        if isinstance(x, (list, tuple)):
            if len(x) == 2:
                return x
            else:
                return (x, x)
        else:
            return (x, x)