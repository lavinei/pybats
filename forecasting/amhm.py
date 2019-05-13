# Analytic Multiscale Holiday Model

# Uses Analytic Multiscale Inference to model holidays separately from the rest of the year

# For now, assume that the standard model is a DBCM... can add in DCMM functionality later
# Would be nice if their syntax were identical

import numpy as np
import pandas as pd


class amhm:
    def __init__(self, mod, nhol, holmod, holidays, X, cal, dates):
        self.mod = mod
        self.nhol = nhol
        self.holmod = holmod
        self.holidays = holidays
        self.X = X
        self.t = 0
        self.cal = cal
        self.dates = dates
        self.holiday_dates = cal.holidays()

    def is_holiday(self, t):
        return self.holiday_dates.contains(self.dates.iloc[t])

    # Update the standard model or the amhm model
    def update(self, **kwargs):
        if self.is_holiday(self.t):
            # forecast = self.mod.forecast_marginal(k=1, nsamps=200, **kwargs, mean_only=True, return_separate=True)
            pois_state_mean, pois_state_var =\
                self.mod.dcmm.pois_mod.forecast_marginal(k = 1,
                                                         X = kwargs.get('X_transaction'),
                                                         state_mean_var=True)
            phi_mu = self.X[self.t,:]
            phi_sigma = np.zeros([self.nhol, self.nhol])
            idx = np.where(phi_mu == 1)
            phi_mu[idx] = pois_state_mean
            phi_sigma[idx, idx] = pois_state_var
            self.holmod.multiscale_update_approx(y_transaction = kwargs.get('y_transaction'), X_transaction=None,
                                                 y_cascade = kwargs.get('y_cascade'), X_cascade=None,
                                                 phi_mu = phi_mu, phi_sigma = phi_sigma,
                                                 excess = kwargs.get('excess'))
            self.mod.update() # Update with no data... this observation is considered missing
        else:
            self.mod.update(**kwargs)
            
        self.t += 1

    # Update the multiscale standard model or the amhm model
    def multiscale_update_approx(self, **kwargs):
        if self.is_holiday(self.t):
            # forecast = self.mod.multiscale_forecast_marginal_approx(k=1, nsamps=200, mean_only=True, return_separate=True, **kwargs)
            pois_state_mean, pois_state_var = \
                self.mod.dcmm.pois_mod.multiscale_forecast_marginal_approx(k = 1,
                                                                           X = kwargs.get('X_transaction'),
                                                                           phi_mu = kwargs.get('phi_mu'),
                                                                           phi_sigma = kwargs.get('phi_sigma'),
                                                                           state_mean_var=True)
            phi_mu = self.X[self.t, :]
            phi_sigma = np.zeros([self.nhol, self.nhol])
            idx = np.where(phi_mu == 1)
            phi_mu[idx] = pois_state_mean
            phi_sigma[idx, idx] = pois_state_var
            self.holmod.multiscale_update_approx(y_transaction=kwargs.get('y_transaction'), X_transaction=None,
                                                 y_cascade=kwargs.get('y_cascade'), X_cascade=None,
                                                 phi_mu=phi_mu, phi_sigma=phi_sigma,
                                                 excess=kwargs.get('excess'))
            self.mod.multiscale_update_approx()  # Update with no data... this observation is considered missing
        else:
            self.mod.multiscale_update_approx(**kwargs)

        self.t += 1

    def forecast_marginal(self, k, **kwargs):

        if self.is_holiday(self.t + k - 1):
            # forecast = self.mod.multiscale_forecast_marginal_approx(k=1, nsamps=200, mean_only=True, return_separate=True, **kwargs)
            pois_state_mean, pois_state_var = \
                self.mod.dcmm.pois_mod.forecast_marginal(k = k,
                                                         X = kwargs.get('X_transaction'),
                                                         state_mean_var=True)
            phi_mu = self.X[self.t + k - 1, :]
            phi_sigma = np.zeros([self.nhol, self.nhol])
            idx = np.where(phi_mu == 1)
            phi_mu[idx] = pois_state_mean
            phi_sigma[idx, idx] = pois_state_var
            return self.holmod.multiscale_forecast_marginal_approx(k = k,
                                                            X_transaction=None,
                                                            X_cascade=None,
                                                            phi_mu=phi_mu, phi_sigma=phi_sigma,
                                                            nsamps = kwargs.get('nsamps'))
        else:
            return self.mod.forecast_marginal(k, **kwargs)

    def multiscale_forecast_marginal_approx(self, k, **kwargs):
        if self.is_holiday(self.t + k - 1):
            # forecast = self.mod.multiscale_forecast_marginal_approx(k=1, nsamps=200, mean_only=True, return_separate=True, **kwargs)
            pois_state_mean, pois_state_var = \
                self.mod.dcmm.pois_mod.multiscale_forecast_marginal_approx(k=k,
                                                                           X=kwargs.get('X_transaction'),
                                                                           phi_mu=kwargs.get('phi_mu'),
                                                                           phi_sigma=kwargs.get('phi_sigma'),
                                                                           state_mean_var=True)
            phi_mu = self.X[self.t + k - 1, :]
            phi_sigma = np.zeros([self.nhol, self.nhol])
            idx = np.where(phi_mu == 1)
            phi_mu[idx] = pois_state_mean
            phi_sigma[idx, idx] = pois_state_var
            return self.holmod.multiscale_forecast_marginal_approx(k=k,
                                                            X_transaction=None,
                                                            X_cascade=None,
                                                            phi_mu=phi_mu, phi_sigma=phi_sigma,
                                                            nsamps=kwargs.get('nsamps'))
        else:
            return self.mod.multiscale_forecast_marginal_approx(k, **kwargs)


    ## Doing the path forecast is challenging ... for now punt back to the marginal forecast
    def forecast_path_approx(self, k, **kwargs):
        """
        NOTE: This is not a real path forecast... challenging to code
        For testing purposes, just doing the marginal forecasts
        :param k:
        :param kwargs:
        :return:
        """
        is_holiday = [self.is_holiday(self.t + j) for j in range(k)]
        if np.any(is_holiday):
            samps = np.zeros([kwargs.get('nsamps'), k])
            for j, is_hol in enumerate(is_holiday):
                samps[:, j] = self.forecast_marginal(k=j + 1,
                                                     X_transaction=kwargs.get('X_transaction')[j, :],
                                                     X_cascade=kwargs.get('X_cascade')[j, :],
                                                     nsamps=kwargs.get('nsamps'))
            return samps
        else:
            return self.mod.forecast_path_approx(k, **kwargs)




        return 0

    def multiscale_forecast_path_approx(self, k, **kwargs):
        """
        NOTE: This is not a real path forecast... challenging to code
        For testing purposes, just doing the marginal forecasts
        :param k:
        :param kwargs:
        :return:
        """
        is_holiday = [self.is_holiday(self.t + j) for j in range(k)]
        if np.any(is_holiday):
            samps = np.zeros([kwargs.get('nsamps'), k])
            for j, is_hol in enumerate(is_holiday):
                samps[:,j] = self.multiscale_forecast_marginal_approx(k = j+1,
                                                                      X_transaction= kwargs.get('X_transaction')[j,:],
                                                                      X_cascade = kwargs.get('X_cascade')[j,:],
                                                                      phi_mu = kwargs.get('phi_mu')[j],
                                                                      phi_sigma = kwargs.get('phi_sigma')[j],
                                                                      nsamps = kwargs.get('nsamps'))
            return samps
        else:
            return self.mod.multiscale_forecast_path_approx(k, **kwargs)


