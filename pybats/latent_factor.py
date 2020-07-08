# Define the class of latent factors (multiscale latent_factors)
from functools import partial

import numpy as np
import pandas as pd
from collections.abc import Iterable
import copy
import pickle

from .seasonal import get_seasonal_effect_fxnl, forecast_weekly_seasonal_factor, forecast_path_weekly_seasonal_factor
from .dbcm import dbcm
from .dcmm import dcmm
from .forecast import forecast_aR, forecast_R_cov
from .dglm import dlm


class latent_factor:
    def __init__(self, mean=None, var=None, forecast_mean=None, forecast_var=None, forecast_cov=None, dates=None, forecast_dates=None,
                 gen_fxn = None, gen_forecast_fxn = None, forecast_path=False):
        self.forecast_mean = pd.Series(forecast_mean, index=forecast_dates)
        self.forecast_var = pd.Series(forecast_var, index=forecast_dates)
        self.forecast_cov = pd.Series(forecast_cov, index=forecast_dates)
        self.mean = pd.Series(mean, index=dates)
        self.var = pd.Series(var, index=dates)
        self.dates = dates
        self.start_date = np.min(dates)
        self.end_date = np.max(dates)
        self.forecast_start_date = np.min(forecast_dates)
        self.forecast_end_date = np.max(forecast_dates)
        self.forecast_dates = forecast_dates
        self.mean_gen = {}
        self.var_gen = {}
        self.forecast_mean_gen = {}
        self.forecast_var_gen = {}
        self.forecast_cov_gen = {}
        if mean is not None:
            if len(dates) != len(mean):
                print('Error: Dates should have the same length as the latent factor')
            if isinstance(mean[0], Iterable):
                self.p = len(mean[0])
            else:
                self.p = 1
            self.k = len(forecast_mean[0]) # forecast length
        self.gen_fxn = gen_fxn
        self.gen_forecast_fxn = gen_forecast_fxn

        self.forecast_path = forecast_path

    def get_lf(self, date):
        return self.mean.loc[date], self.var.loc[date]

    def get_lf_forecast(self, date):
        if self.forecast_path:
            return self.forecast_mean.loc[date], self.forecast_var.loc[date], self.forecast_cov.loc[date]
        else:
            return self.forecast_mean.loc[date], self.forecast_var.loc[date]

    def generate_lf(self, date, **kwargs):
        m, v = self.gen_fxn(date, **kwargs)
        self.mean_gen.update({date:m})
        self.var_gen.update({date:v})
        # m = pd.Series({date:m})
        # v = pd.Series({date:v})
        # self.mean = self.mean.append(m)
        # self.var = self.var.append(v)

    def generate_lf_forecast(self, date, **kwargs):
        if self.forecast_path:
            m, v, cov = self.gen_forecast_fxn(date, forecast_path=self.forecast_path, **kwargs)
            self.forecast_cov_gen.update({date:cov})
        else:
            m, v = self.gen_forecast_fxn(date, forecast_path=self.forecast_path, **kwargs)

        self.forecast_mean_gen.update({date:m})
        self.forecast_var_gen.update({date:v})
        # m = pd.Series({date:m})
        # v = pd.Series({date:v})
        # self.forecast_mean = self.forecast_mean.append(m)
        # self.forecast_var = self.forecast_var.append(v)

    def append_lf(self):
        self.mean = self.mean.append(pd.Series(self.mean_gen))
        self.var = self.var.append(pd.Series(self.var_gen))
        self.mean_gen = {}
        self.var_gen = {}
        if isinstance(self.mean.head().values[0], Iterable):
            self.p = len(self.mean.head().values[0])
        else:
            self.p = 1
        self.start_date = np.min(self.mean.index.values)
        self.end_date = np.max(self.mean.index.values)
        self.dates = self.mean.index

    def append_lf_forecast(self):
        self.forecast_mean = self.forecast_mean.append(pd.Series(self.forecast_mean_gen))
        self.forecast_var = self.forecast_var.append(pd.Series(self.forecast_var_gen))
        self.forecast_cov = self.forecast_cov.append(pd.Series(self.forecast_cov_gen))
        self.forecast_mean_gen = {}
        self.forecast_var_gen = {}
        self.forecast_cov_gen = {}
        if isinstance(self.forecast_mean.head().values[0], Iterable):
            self.k = len(self.forecast_mean.head().values[0])
        else:
            self.k = 1
        self.forecast_start_date = np.min(self.forecast_mean.index.values)
        self.forecast_end_date = np.max(self.forecast_mean.index.values)
        self.forecast_dates = self.forecast_mean.index

    def copy(self):
        newlf = copy.deepcopy(self)

        return newlf

    def save(self, filename):
        file = open(filename, "wb")
        pickle.dump(self, file=file)


class multi_latent_factor(latent_factor):
    def __init__(self, latent_factors):
        """
        :param latent_factors: Tuple that contains only objects of class 'latent_factor'
        """
        self.n_lf = len(latent_factors)
        self.p = np.sum([lf.p for lf in latent_factors])
        self.k = np.min([lf.k for lf in latent_factors])
        self.latent_factors = latent_factors

        # initialize matrices that filled in when 'get_lf' and 'get_lf_forecast' are called
        self.mean = np.zeros(self.p)
        self.var = np.zeros([self.p, self.p])
        self.forecast_mean = [np.zeros(self.p) for k in range(self.k)]
        self.forecast_var = [np.zeros([self.p, self.p]) for k in range(self.k)]
        self.forecast_cov = [np.zeros([self.p, self.p, k]) for k in range(1, self.k)]

        # Set the start and end dates
        start_date = np.max([lf.start_date for lf in latent_factors])
        end_date = np.min([lf.end_date for lf in latent_factors])
        self.dates = pd.date_range(start_date, end_date)

        # Set the start and end forecast dates
        forecast_start_date = np.max([lf.forecast_start_date for lf in latent_factors])
        forecast_end_date = np.min([lf.forecast_end_date for lf in latent_factors])
        self.forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)

        if np.all([lf.forecast_path for lf in self.latent_factors]):
            self.forecast_path = True
        else:
            self.forecast_path = False

    def get_lf(self, date):
        idx = 0
        for lf in self.latent_factors:
            m, v = lf.get_lf(date)
            self.mean[idx:idx + lf.p] = m
            self.var[idx:idx + lf.p, idx:idx + lf.p] = v
            idx += lf.p

        return self.mean, self.var

    def get_lf_forecast(self, date):
        idx = 0
        if self.forecast_path:
            for lf in self.latent_factors:

                f_m, f_v, f_c = lf.get_lf_forecast(date)
                for k, [m, v] in enumerate(zip(f_m, f_v)):
                    self.forecast_mean[k][idx:idx + lf.p] = m
                    self.forecast_var[k][idx:idx + lf.p, idx:idx + lf.p] = v
                    if k > 0:
                        self.forecast_cov[k - 1][idx:idx + lf.p, idx:idx + lf.p, :] = f_c[k-1]
                idx += lf.p
            return self.forecast_mean, self.forecast_var, self.forecast_cov

        else:
            for lf in self.latent_factors:
                f_m, f_v = lf.get_lf_forecast(date)
                for k, [m, v] in enumerate(zip(f_m, f_v)):
                    self.forecast_mean[k][idx:idx + lf.p] = m
                    self.forecast_var[k][idx:idx + lf.p, idx:idx + lf.p] = v
                idx += lf.p
            return self.forecast_mean, self.forecast_var


    def copy(self):

        new_lfs = []
        for lf in self.latent_factors:

            new_lfs.append(lf.copy())

        return multi_latent_factor(new_lfs)

    def add_latent_factor(self, latent_factor):
        """
        :param latent_factor: A new latent factor to be added to the multi_latent_factor
        :return:
        """
        # Append the new latent_factor on
        self.latent_factors.append(latent_factor)

        self.n_lf = len(self.latent_factors)
        self.p = np.sum([lf.p for lf in self.latent_factors])
        self.k = np.min([lf.k for lf in self.latent_factors])

        # initialize matrices that filled in when 'get_lf' and 'get_lf_forecast' are called
        self.mean = np.zeros(self.p)
        self.var = np.zeros([self.p, self.p])
        self.forecast_mean = [np.zeros(self.p) for k in range(self.k)]
        self.forecast_var = [np.zeros([self.p, self.p]) for k in range(self.k)]
        self.forecast_cov = [np.zeros([self.p, self.p, k]) for k in range(1, self.k)]

        # Set the start and end dates
        start_date = np.max([lf.start_date for lf in self.latent_factors])
        end_date = np.min([lf.end_date for lf in self.latent_factors])
        self.dates = pd.date_range(start_date, end_date)

        # Set the start and end forecast dates
        forecast_start_date = np.max([lf.forecast_start_date for lf in self.latent_factors])
        forecast_end_date = np.min([lf.forecast_end_date for lf in self.latent_factors])
        self.forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)

    def drop_latent_factor(self, idx):
        """
        :param idx: Index of the latent factor to drop
        :return:
        """
        # Append the new latent_factor on
        self.latent_factors.pop(idx)

        self.n_lf = len(self.latent_factors)
        self.p = np.sum([lf.p for lf in self.latent_factors])
        self.k = np.min([lf.k for lf in self.latent_factors])

        # initialize matrices that filled in when 'get_lf' and 'get_lf_forecast' are called
        self.mean = np.zeros(self.p)
        self.var = np.zeros([self.p, self.p])
        self.forecast_mean = [np.zeros(self.p) for k in range(self.k)]
        self.forecast_var = [np.zeros([self.p, self.p]) for k in range(self.k)]
        self.forecast_cov = [np.zeros([self.p, self.p, k]) for k in range(1, self.k)]

        # Set the start and end dates
        start_date = np.max([lf.start_date for lf in self.latent_factors])
        end_date = np.min([lf.end_date for lf in self.latent_factors])
        self.dates = pd.date_range(start_date, end_date)

        # Set the start and end forecast dates
        forecast_start_date = np.max([lf.forecast_start_date for lf in self.latent_factors])
        forecast_end_date = np.min([lf.forecast_end_date for lf in self.latent_factors])
        self.forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)

    def save(self, filename):
        file = open(filename, "wb")
        pickle.dump(self, file=file)


#### A number of common latent factor functions
def hol_fxn(date, mod, X, **kwargs):
    is_hol = np.any(X[-mod.nhol:] != 0)
    mean = np.zeros(mod.nhol)
    var = np.zeros([mod.nhol, mod.nhol])
    if is_hol:
        idx = np.where(X[-mod.nhol:] != 0)[0][0]
        mean[idx] = X[-mod.nhol:] @ mod.m[mod.ihol]
        var[idx, idx] = X[-mod.nhol:] @ mod.C[np.ix_(mod.ihol, mod.ihol)] @ X[-mod.nhol:]

    return mean, var

def hol_forecast_fxn(date, mod, X, k, horizons, forecast_path=False, **kwargs):

    future_holiday_eff = list(map(lambda X, k: forecast_holiday_effect_dlm(mod, X, k),
                                  X[:, -mod.nhol:], horizons))
    hol_mean = [np.zeros(mod.nhol) for h in range(k)]
    hol_var = [np.zeros([mod.nhol, mod.nhol]) for h in range(k)]

    if forecast_path:
        hol_cov = [np.zeros([mod.nhol, mod.nhol, h]) for h in range(1, k)]
        nonzero_holidays = {}

        for h in range(k):
            if future_holiday_eff[h][0] != 0:
                idx = np.where(X[h, -mod.nhol:] != 0)[0][0]
                hol_mean[h][idx] = future_holiday_eff[h][0]
                hol_var[h][idx, idx] = future_holiday_eff[h][1]

                for j, idx_j in nonzero_holidays.items():
                    hol_cov[h-1][idx, idx_j, j] = hol_cov[h-1][idx_j, idx, j] = X[j, -mod.nhol:] @ forecast_R_cov(mod, j, h)[np.ix_(mod.ihol, mod.ihol)] @ X[h, -mod.nhol:].T

                nonzero_holidays.update({h:idx})

        return hol_mean, hol_var, hol_cov

    else:
        for h in range(k):
            if future_holiday_eff[h][0] != 0:
                idx = np.where(X[h, -mod.nhol:] != 0)[0][0]
                hol_mean[h][idx] = future_holiday_eff[h][0]
                hol_var[h][idx, idx] = future_holiday_eff[h][1]


        return hol_mean, hol_var




hol_lf = latent_factor(gen_fxn = hol_fxn, gen_forecast_fxn=hol_forecast_fxn)


def Y_fxn(date, mod, Y, **kwargs):
    return Y, 0


def Y_forecast_fxn(date, mod, X, k, nsamps, horizons, forecast_path = False, **kwargs):
    #
    # Y_mean = [f.mean() for f in forecast]

    if forecast_path:
        # if isinstance(mod, dlm):
        #     mean, var = mod.forecast_path(k=k, X=X, nsamps=nsamps, mean_var = True)
        #     Y_mean = [m for m in mean]
        #     Y_var = [v for v in var.diagonal()]
        #     Y_cov = [var[h,:h].reshape(1,1,h) for h in range(1, k)]
        # else:
            # print('error')
        forecast = mod.forecast_path(k=k, X=X, nsamps=nsamps)
        Y_mean = [m for m in forecast.mean(axis=0)]
        cov = np.cov(forecast, rowvar=False)
        Y_var = [v for v in forecast.var(axis=0)]
        Y_cov = [cov[h,:h].reshape(1, 1, -1) for h in range(1, k)]

        return Y_mean, Y_var, Y_cov

    else:
        forecast = list(map(lambda X, k: mod.forecast_marginal(k=k, X=X, nsamps=nsamps),
                            X,
                            horizons))
        Y_mean = list(map(lambda X, k: mod.forecast_marginal(k=k, X=X, mean_only=True),
                          X,
                          horizons))
        Y_var = [f.var() for f in forecast]
        return Y_mean, Y_var


Y_lf = latent_factor(gen_fxn = Y_fxn, gen_forecast_fxn = Y_forecast_fxn)

# This is a good idea, but note that it does not work with the set analysis fxns,
# This forecast must happen BEFORE updating, not after
def Y_update_via_forecast_fxn(date, mod, X, nsamps=200, **kwargs):
    mean = mod.forecast_marginal(k=1, X=X, mean_only=True)
    forecast = mod.forecast_marginal(k=1, X=X)
    return mean, forecast.var()


Y_forecast_lf = latent_factor(gen_fxn = Y_update_via_forecast_fxn, gen_forecast_fxn = Y_forecast_fxn)


def seas_weekly_fxn(date, mod, **kwargs):
    period = 7
    seas_idx = np.where(np.array(mod.seasPeriods) == 7)[0][0]
    today = date.dayofweek
    m, v = get_seasonal_effect_fxnl(mod.L[seas_idx], mod.m, mod.C, mod.iseas[seas_idx])
    weekly_seas_mean = np.zeros(period)
    weekly_seas_var = np.zeros([period, period])
    weekly_seas_mean[today] = m
    weekly_seas_var[today, today] = v

    return weekly_seas_mean, weekly_seas_var


def seas_weekly_forecast_fxn(date, mod, k, horizons, forecast_path=False, **kwargs):
    period = 7
    today = date.dayofweek

    if isinstance(mod, dlm):
        rt = mod.n / (mod.n-2)
    else:
        rt = 1

    if forecast_path:
        weekly_seas_mean, weekly_seas_var, weekly_seas_cov = forecast_path_weekly_seasonal_factor(mod, k, today, period)
        return weekly_seas_mean,  [rt * wsv for wsv in weekly_seas_var], [rt * wsc for wsc in weekly_seas_cov]
    else:

        # Place the weekly seasonal factor into the correct spot in a length 7 vector
        future_weekly_seas = list(map(lambda k: forecast_weekly_seasonal_factor(mod, k=k),
                                      horizons))
        weekly_seas_mean = [np.zeros(period) for i in range(k)]
        weekly_seas_var = [np.zeros([period, period]) for i in range(k)]
        for i in range(k):
            day = (today + i) % period
            weekly_seas_mean[i][day] = future_weekly_seas[i][0]
            weekly_seas_var[i][day, day] = future_weekly_seas[i][1]

        return weekly_seas_mean, [rt * wsv for wsv in weekly_seas_var]


seas_weekly_lf = latent_factor(gen_fxn = seas_weekly_fxn, gen_forecast_fxn=seas_weekly_forecast_fxn)


def pois_coef_fxn(date, mod, idx = None, **kwargs):
    if type(mod) == dbcm:
        if idx is None:
            idx = np.arange(0, len(mod.dcmm.pois_mod.m))

        return mod.dcmm.pois_mod.m[idx].copy().reshape(-1), mod.dcmm.pois_mod.C[np.ix_(idx, idx)].copy()
    if type(mod) == dcmm:
        if idx is None:
            idx = np.arange(0, len(mod.pois_mod.m))

        return mod.pois_mod.m[idx].copy().reshape(-1), mod.pois_mod.C[np.ix_(idx, idx)].copy()


def pois_coef_forecast_fxn(date, mod, k, idx=None, **kwargs):
    if type(mod) == dbcm:
        if idx is None:
            idx = np.arange(0, len(mod.dcmm.pois_mod.m))

        pois_coef_mean = []
        pois_coef_var = []
        for j in range(1, k + 1):
            a, R = forecast_aR(mod.dcmm.pois_mod, j)
            pois_coef_mean.append(a[idx].copy().reshape(-1))
            pois_coef_var.append(R[np.ix_(idx, idx)].copy())
        return pois_coef_mean, pois_coef_var
    if type(mod) == dcmm:
        if idx is None:
            idx = np.arange(0, len(mod.pois_mod.m))

        pois_coef_mean = []
        pois_coef_var = []
        for j in range(1, k + 1):
            a, R = forecast_aR(mod.pois_mod, j)
            pois_coef_mean.append(a[idx].copy().reshape(-1))
            pois_coef_var.append(R[np.ix_(idx, idx)].copy())
        return pois_coef_mean, pois_coef_var


pois_coef_lf = latent_factor(gen_fxn = pois_coef_fxn, gen_forecast_fxn=pois_coef_forecast_fxn)


def bern_coef_fxn(date, mod, idx = None, **kwargs):
    if type(mod) == dbcm:
        if idx is None:
            idx = np.arange(0, len(mod.dcmm.bern_mod.m))

        return mod.dcmm.bern_mod.m[idx].copy().reshape(-1), mod.dcmm.bern_mod.C[np.ix_(idx, idx)].copy()
    if type(mod) == dcmm:
        if idx is None:
            idx = np.arange(0, len(mod.bern_mod.m))

        return mod.bern_mod.m[idx].copy().reshape(-1), mod.bern_mod.C[np.ix_(idx, idx)].copy()


def bern_coef_forecast_fxn(date, mod, k, idx = None, **kwargs):
    if type(mod) == dbcm:
        if idx is None:
            idx = np.arange(0, len(mod.dcmm.bern_mod.m))

        bern_coef_mean = []
        bern_coef_var = []
        for j in range(1, k + 1):
            a, R = forecast_aR(mod.dcmm.bern_mod, j)
            bern_coef_mean.append(a[idx].copy().reshape(-1))
            bern_coef_var.append(R[np.ix_(idx, idx)].copy())
        return bern_coef_mean, bern_coef_var

    if type(mod) == dcmm:
        if idx is None:
            idx = np.arange(0, len(mod.bern_mod.m))

        bern_coef_mean = []
        bern_coef_var = []
        for j in range(1, k + 1):
            a, R = forecast_aR(mod.dcmm.bern_mod, j)
            bern_coef_mean.append(a[idx].copy().reshape(-1))
            bern_coef_var.append(R[np.ix_(idx, idx)].copy())
        return bern_coef_mean, bern_coef_var


bern_coef_lf = latent_factor(gen_fxn=bern_coef_fxn, gen_forecast_fxn=bern_coef_forecast_fxn)


def dlm_coef_fxn(date, mod, idx = None, **kwargs):
    if idx is None:
        idx = np.arange(0, len(mod.m))

    return mod.m[idx].copy().reshape(-1), mod.C[np.ix_(idx, idx)].copy()


def dlm_coef_forecast_fxn(date, mod, k, idx=None, forecast_path=False, **kwargs):
    if idx is None:
        idx = np.arange(0, len(mod.m))

    p = len(idx)

    dlm_coef_mean = []
    dlm_coef_var = []
    if forecast_path:
        dlm_coef_cov = [np.zeros([p, p, h]) for h in range(1, k)]

    for j in range(1, k + 1):
        a, R = forecast_aR(mod, j)
        dlm_coef_mean.append(a[idx].copy().reshape(-1))
        dlm_coef_var.append(R[np.ix_(idx, idx)].copy())

        if forecast_path:
            if j > 1:
                for i in range(1, j):
                    dlm_coef_cov[j-2][:,:,i-1] = forecast_R_cov(mod, i, j)[np.ix_(idx, idx)]

    if forecast_path:
        return dlm_coef_mean, dlm_coef_var, dlm_coef_cov
    else:
        return dlm_coef_mean, dlm_coef_var


dlm_coef_lf = latent_factor(gen_fxn = dlm_coef_fxn, gen_forecast_fxn=dlm_coef_forecast_fxn)

def dlm_dof_fxn(date, mod, **kwargs):
    return mod.n, 0


def dlm_dof_forecast_fxn(date, mod, k, **kwargs):
    return [mod.n for x in range(k)], [0 for x in range(k)]

dlm_dof_lf = latent_factor(gen_fxn = dlm_dof_fxn, gen_forecast_fxn=dlm_dof_forecast_fxn)


def copy_fxn(date, latent_factor):
    s = latent_factor.get_lf(date)
    return copy.deepcopy(s[0]), copy.deepcopy(s[1])


def copy_forecast_fxn(date, latent_factor):
    means = []
    vars = []
    ms, vs = latent_factor.get_lf_forecast(date)
    for h in range(latent_factor.k):
        means.append(copy.deepcopy(ms[h]))
        vars.append(copy.deepcopy(vs[h]))
    return means, vars


def merge_fxn(date, latent_factors, **kwargs):
    if latent_factors[0].p == 1:
        m = np.array([float(lf.get_lf(date)[0]) for lf in latent_factors])
        v = np.array([float(lf.get_lf(date)[1]) for lf in latent_factors])
        p = 1 / v
        return np.sum(m * p) / np.sum(p), 1 / np.sum(p)
    else:
        ms = [lf.get_lf(date)[0] for lf in latent_factors]
        vs = [lf.get_lf(date)[1] for lf in latent_factors]
        ps = [np.linalg.inv(v) for v in vs]
        m = np.sum([p @ m.reshape(-1,1) for m, p in zip(ms, ps)], axis=0)
        v = np.linalg.inv(np.sum(ps, axis=0))
        mean = v @ m
        return mean.reshape(-1), v


def merge_forecast_fxn(date, latent_factors, **kwargs):
    k = np.min([lf.k for lf in latent_factors])
    lf_mean = []
    lf_var = []
    if latent_factors[0].p == 1:
        ms, vs = list(zip(*[lf.get_lf_forecast(date) for lf in latent_factors]))
        for h in range(k):
            m = np.array([float(m[h]) for m in ms])
            v = np.array([float(v[h]) for v in vs])
            p = 1 / v
            lf_mean.append(np.sum(m * p) / np.sum(p))
            lf_var.append(1 / np.sum(p))
        return lf_mean, lf_var
    else:
        ms, vs = list(zip(*[lf.get_lf_forecast(date) for lf in latent_factors]))
        ps = [[np.linalg.inv(var) for var in v] for v in vs]
        for h in range(k):
            m = np.sum([p[h] @ m[h].reshape(-1, 1) for m, p in zip(ms, ps)], axis=0)
            v = np.linalg.inv(np.sum([p[h] for p in ps], axis=0))
            mean = v @ m
            lf_mean.append(mean.reshape(-1))
            lf_var.append(v)
        return lf_mean, lf_var


def merge_latent_factors(latent_factors):
    """
    :param latent_factors: list of the same latent factor from different sources to be combined into 1 using precision weighted averaging
    :return: A single latent factor
    """
    # Set the start and end dates
    start_date = np.min([lf.start_date for lf in latent_factors])
    end_date = np.max([lf.end_date for lf in latent_factors])
    dates = pd.date_range(start_date, end_date)

    # Set the start and end forecast dates
    forecast_start_date = np.min([lf.forecast_start_date for lf in latent_factors])
    forecast_end_date = np.max([lf.forecast_end_date for lf in latent_factors])
    forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)

    # Create a new latent factor
    merged_lf = latent_factor(gen_fxn = merge_fxn,
                              gen_forecast_fxn = merge_forecast_fxn)

    for date in dates:
        merged_lf.generate_lf(date, latent_factors=[lf for lf in latent_factors if lf.dates.isin([date]).any()])

    for date in forecast_dates:
        merged_lf.generate_lf_forecast(date, latent_factors=[lf for lf in latent_factors if lf.forecast_dates.isin([date]).any()])

    merged_lf.append_lf()
    merged_lf.append_lf_forecast()

    return merged_lf


def merge_lf_with_predictor(latent_factor, X, X_dates):
    """
    Function to modify a latent factor by multiplying it by a known predictor. Example of latent factor is
     the coefficient on effect of price from an external model, while the price itself is a known predictor.

    :param X: A known predictor
    :param X_dates: Dates associated with the known predictor
    :return:
    """

    newlf = latent_factor.copy()

    X = pd.DataFrame(X, index=X_dates)
    if latent_factor.p == 1:

        for date in newlf.dates:
            if X_dates.isin([date]).any():
                newlf.mean.loc[date] *= X.loc[date].values
                newlf.var.loc[date] *= (X.loc[date].values ** 2)
            else:
                newlf.mean.drop(date, inplace=True)
                newlf.var.drop(date, inplace=True)

        for date in newlf.forecast_dates:
            if X_dates.isin([date]).any():

                # m = newlf.forecast_mean.loc[date]
                # v = newlf.forecast_var.loc[date]
                for h in range(newlf.k):
                    newlf.forecast_mean.loc[date][h] *= X.loc[date + pd.DateOffset(days=h)].values
                    newlf.forecast_var.loc[date][h] *= (X.loc[date + pd.DateOffset(days=h)].values ** 2)
                # newlf.forecast_mean.loc[date] = m
                # newlf.forecast_var.loc[date] = v

                if newlf.forecast_path:
                    # c = newlf.forecast_cov.loc[date]
                    for h in range(1, newlf.k):
                        for j in range(h):
                            newlf.forecast_cov.loc[date][h-1][:,:,j] *= X.loc[date + pd.DateOffset(days=j)].values * X.loc[date + pd.DateOffset(days=h)].values
                    # newlf.forecast_cov.loc[date] = c
            else:
                newlf.forecast_mean.drop(date, inplace=True)
                newlf.forecast_var.drop(date, inplace=True)

                if newlf.forecast_path:
                    newlf.forecast_cov.drop(date, inplace=True)


    else:
        for date in newlf.dates:
            if X_dates.isin([date]).any():
                newlf.mean.loc[date] *= X.loc[date].values
                newlf.var.loc[date] *= X.loc[date].values.reshape(-1,1) @ X.loc[date].values.reshape(1,-1)
            else:
                newlf.mean.drop(date, inplace=True)
                newlf.var.drop(date, inplace=True)

        for date in newlf.forecast_dates:
            if X_dates.isin([date]).any():
                # m = newlf.forecast_mean.loc[date]
                # v = newlf.forecast_var.loc[date]
                for h in range(newlf.k):
                    newlf.forecast_mean.loc[date][h] *= X.loc[date + pd.DateOffset(days=h)].values
                    newlf.forecast_var.loc[date][h] *= X.loc[date + pd.DateOffset(days=h)].values.reshape(-1,1) @ X.loc[date + pd.DateOffset(days=h)].values.reshape(1,-1)
                # newlf.forecast_mean.loc[date] = m
                # newlf.forecast_var.loc[date] = v

                if newlf.forecast_path:
                    # c = newlf.forecast_cov.loc[date]
                    for h in range(1, newlf.k):
                        for j in range(h):
                            newlf.forecast_cov.loc[date][h-1][:,:,j] *= X.loc[date + pd.DateOffset(days=j)].values.reshape(-1,1) @ X.loc[date + pd.DateOffset(days=h)].values.reshape(1,-1)

                    # newlf.forecast_cov.loc[date] = c

            else:
                newlf.forecast_mean.drop(date, inplace=True)
                newlf.forecast_var.drop(date, inplace=True)

                if newlf.forecast_path:
                    newlf.forecast_cov.drop(date, inplace=True)

    return newlf


def pct_chg_from_lf_avg(latent_factor, window=10):

    def pct_chg_gen(date, ma, lf):
        m = 100 * (lf.mean.loc[date] - ma.loc[date]) / ma.loc[date]
        v = lf.var.loc[date] * (100 / ma.loc[date]) ** 2
        return m, v

    def pct_chg_gen_forecast(date, ma, k, lf):
        m = []
        v = []
        for h in range(lf.k):
            m.append(100 * ((lf.forecast_mean.loc[date][h] - ma.loc[date]) / ma.loc[date]))
            v.append(lf.forecast_var.loc[date][h] * (100 / ma.loc[date]) ** 2)

        return m, v


    ma = latent_factor.mean.rolling(window=window, min_periods=1).mean()
    ma = pd.Series(ma, index=latent_factor.dates)

    newlf = latent_factor(gen_fxn = partial(pct_chg_gen, lf=latent_factor, ma=ma), gen_forecast_fxn=partial(pct_chg_gen_forecast, lf=latent_factor, ma=ma, k=latent_factor.k))

    for date in latent_factor.dates:
        newlf.generate_lf(date)

    for date in latent_factor.forecast_dates:
        newlf.generate_lf_forecast(date)

    newlf.append_lf()
    newlf.append_lf_forecast()

    return newlf


def load_latent_factor(filename):
    file = open(filename, 'rb')
    return pickle.load(file)


def forecast_holiday_effect_dlm(mod, X, k):
    a, R = forecast_aR(mod, k)

    mean = X.T @ a[mod.ihol]
    var = (mod.n / (mod.n - 2)) * (X.T @ R[np.ix_(mod.ihol, mod.ihol)] @ X + mod.s)
    return mean, var

# def forecast_dlm_path(mod, k, X = None, nsamps = 1, mean_var=False):
#     samps = np.zeros([nsamps, k])
#     F = np.copy(mod.F)
#
#     if not mod.discount_forecast:
#         mean = np.zeros([k])
#         scale = np.zeros([k,k])
#
#         Flist = [None for x in range(k)]
#         Rlist = [None for x in range(k)]
#
#         for i in range(k):
#
#             # Evolve to the prior at time t + i + 1
#             a, R = forecast_aR(mod, i + 1)
#
#             Rlist[i] = R
#
#             # Plug in the correct F values
#             F = update_F(mod, X[i, :], F=F)
#             # if mod.nregn > 0:
#             #     F[mod.iregn] = X[i,:].reshape(mod.nregn,1)
#
#             Flist[i] = np.copy(F)
#
#             # Find lambda mean and var
#             ft, qt = mod.get_mean_and_var(F, a, R)
#             mean[i] = ft
#             scale[i, i] = qt
#
#             # Find covariances with previous lambda values
#             for j in range(i):
#                 # Covariance matrix between the state vector at times j, i, i > j
#                 cov_ij = np.linalg.matrix_power(mod.G, i - j) @ Rlist[j]
#                 # Covariance between lambda at times j, i
#                 scale[j, i] = scale[i, j] = Flist[j].T @ cov_ij @ Flist[i]
#
#         if mean_var:
#             return mean, mod.n/(mod.n - 2) * scale
#
#         return multivariate_t(mean, scale, mod.n, nsamps) # if discounting the forecast, then multivariate t is wrong, b/c each horizon has a different df
#     else:
#         print('Cannot get an exact multivariate t distribution because the degrees of freedom differ')

#
# def hol_forecast_fxn_sample(date, mod, X, k, horizons, forecast_path=False, nsamps=1, **kwargs):
#
#     sample_list = []
#     X_hol = X[: -mod.nhol:]
#
#     if forecast_path:
#
#
#         hol_cov = [np.zeros([mod.nhol, mod.nhol, h]) for h in range(1, k)]
#         nonzero_holidays = {}
#
#         for h in range(k):
#             if future_holiday_eff[h][0] != 0:
#                 idx = np.where(X[h, -mod.nhol:] != 0)[0][0]
#                 hol_mean[h][idx] = future_holiday_eff[h][0]
#                 hol_var[h][idx, idx] = future_holiday_eff[h][1]
#
#                 for j, idx_j in nonzero_holidays.items():
#                     hol_cov[h-1][idx, idx_j, j] = hol_cov[h-1][idx_j, idx, j] = X[j, -mod.nhol:] @ forecast_R_cov(mod, j, h)[np.ix_(mod.ihol, mod.ihol)] @ X[h, -mod.nhol:].T
#
#                 nonzero_holidays.update({h:idx})
#
#
#     else:
#         future_holiday_eff = list(map(lambda X, k: forecast_holiday_effect_dlm(mod, X, k),
#                                       X[:, -mod.nhol:], horizons))
#         hol_mean = [np.zeros(mod.nhol) for h in range(k)]
#         hol_var = [np.zeros([mod.nhol, mod.nhol]) for h in range(k)]
#
#         samps = np.zeros([nsamps, mod.nhol])
#
#         for h in range(k):
#             a, R = forecast_aR(mod, h)
#             if not np.all(X_hol[h,:] == 0):
#                 mean = X_hol[h,:].T @ a[mod.ihol]
#                 var = (X_hol[h,:].T @ R[np.ix_(mod.ihol, mod.ihol)] @ X + mod.s)
#
#             if future_holiday_eff[h][0] != 0:
#                 idx = np.where(X[h, -mod.nhol:] != 0)[0][0]
#                 hol_mean[h][idx] = future_holiday_eff[h][0]
#                 hol_var[h][idx, idx] = future_holiday_eff[h][1]
#                 samps = mod.simulate(mean = hol_mean[h], var = hol_var[h], size=[nsamps])
#
#             sample_list.append(samps)
#
#     return sample_list


#
# def hol_fxn_sample(date, mod, X, nsamps, **kwargs):
#     is_hol = np.any(X[-mod.nhol:] != 0)
#     samps = np.zeros([nsamps, mod.nhol])
#     if is_hol:
#         idx = np.where(X[-mod.nhol:] != 0)[0][0]
#         mean = X[-mod.nhol:] @ mod.m[mod.ihol]
#         var = X[-mod.nhol:] @ mod.C[np.ix_(mod.ihol, mod.ihol)] @ X[-mod.nhol:]
#         samps[:,idx] = mod.simulate(mean=mean, var=var, nsamps=nsamps)
#
#     return samps
#
