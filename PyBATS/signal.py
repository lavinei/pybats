# Define the class of latent factors (multiscale signals)
import numpy as np
import pandas as pd
from collections.abc import Iterable
import copy

from .multiscale import forecast_holiday_effect
from .seasonal import get_seasonal_effect_fxnl, forecast_weekly_seasonal_factor
from .dbcm import dbcm
from .dcmm import dcmm
from .forecast import forecast_aR

class signal:
    def __init__(self, mean=None, var=None, forecast_mean=None, forecast_var=None, dates=None, forecast_dates=None,
                 gen_fxn = None, gen_forecast_fxn = None):
        self.forecast_mean = pd.Series(forecast_mean, index=forecast_dates)
        self.forecast_var = pd.Series(forecast_var, index=forecast_dates)
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
        if mean is not None:
            if len(dates) != len(mean):
                print('Error: Dates should have the same length as the signal')
            if isinstance(mean[0], Iterable):
                self.p = len(mean[0])
            else:
                self.p = 1
            self.k = len(forecast_mean[0]) # forecast length
        self.gen_fxn = gen_fxn
        self.gen_forecast_fxn = gen_forecast_fxn

    def get_signal(self, date):
        return self.mean.loc[date], self.var.loc[date]

    def get_forecast_signal(self, date):
        return self.forecast_mean.loc[date], self.forecast_var.loc[date]

    def generate_signal(self, date, **kwargs):
        m, v = self.gen_fxn(date, **kwargs)
        self.mean_gen.update({date:m})
        self.var_gen.update({date:v})
        # m = pd.Series({date:m})
        # v = pd.Series({date:v})
        # self.mean = self.mean.append(m)
        # self.var = self.var.append(v)

    def generate_forecast_signal(self, date, **kwargs):
        m, v = self.gen_forecast_fxn(date, **kwargs)
        self.forecast_mean_gen.update({date:m})
        self.forecast_var_gen.update({date:v})
        # m = pd.Series({date:m})
        # v = pd.Series({date:v})
        # self.forecast_mean = self.forecast_mean.append(m)
        # self.forecast_var = self.forecast_var.append(v)

    def append_signal(self):
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

    def append_forecast_signal(self):
        self.forecast_mean = self.forecast_mean.append(pd.Series(self.forecast_mean_gen))
        self.forecast_var = self.forecast_var.append(pd.Series(self.forecast_var_gen))
        self.forecast_mean_gen = {}
        self.forecast_var_gen = {}
        if isinstance(self.forecast_mean.head().values[0], Iterable):
            self.k = len(self.forecast_mean.head().values[0])
        else:
            self.k = 1
        self.forecast_start_date = np.min(self.forecast_mean.index.values)
        self.forecast_end_date = np.max(self.forecast_mean.index.values)
        self.forecast_dates = self.forecast_mean.index

    def copy(self):
        # newsig = signal(gen_fxn=copy_fxn, gen_forecast_fxn=copy_forecast_fxn)
        # if self.dates is None:
        #     dates = None
        #     mean = None
        #     var = None
        # else:
            # for date in self.dates:
            #     newsig.generate_signal(date, signal = self)

            # dates = self.dates.copy()
            # mean = self.mean.copy().values
            # var = self.var.copy().values

        # if self.forecast_dates is None:
        #     forecast_dates = None
        #     forecast_mean = None
        #     forecast_var = None
        # else:
            # for date in self.dates:
            #     newsig.generate_forecast_signal(date, signal=self)
            # forecast_dates = self.forecast_dates.copy()
            # forecast_mean = self.forecast_mean.copy().values
            # forecast_var = self.forecast_var.copy().values


        # newsig = signal(mean=mean, var=var,
        #                 forecast_mean=forecast_mean, forecast_var=forecast_var,
        #                 dates= dates, forecast_dates=forecast_dates,
        #                 gen_fxn=self.gen_fxn, gen_forecast_fxn=self.gen_forecast_fxn)

        # newsig.append_signal()
        # newsig.append_forecast_signal()

        newsig = copy.deepcopy(self)

        return newsig



class multisignal(signal):
    def __init__(self, signals):
        """
        :param signals: Tuple that contains only objects of class 'signal'
        """
        self.nsignals = len(signals)
        self.p = np.sum([sig.p for sig in signals])
        self.k = np.min([sig.k for sig in signals])
        self.signals = signals

        # initialize matrices that filled in when 'get_signal' and 'get_forecast_signal' are called
        self.mean = np.zeros(self.p)
        self.var = np.zeros([self.p, self.p])
        self.forecast_mean = [np.zeros(self.p) for k in range(self.k)]
        self.forecast_var = [np.zeros([self.p, self.p]) for k in range(self.k)]
        self.forecast_covar = [np.zeros([self.p, self.p]) for k in range(self.k)]

        # Set the start and end dates
        start_date = np.max([sig.start_date for sig in signals])
        end_date = np.min([sig.end_date for sig in signals])
        self.dates = pd.date_range(start_date, end_date)

        # Set the start and end forecast dates
        forecast_start_date = np.max([sig.forecast_start_date for sig in signals])
        forecast_end_date = np.min([sig.forecast_end_date for sig in signals])
        self.forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)

    def get_signal(self, date):
        idx = 0
        for sig in self.signals:
            m, v = sig.get_signal(date)
            self.mean[idx:idx + sig.p] = m
            self.var[idx:idx + sig.p, idx:idx + sig.p] = v
            idx += sig.p

        return self.mean, self.var

    def get_forecast_signal(self, date):
        idx = 0
        for sig in self.signals:
            f_m, f_v = sig.get_forecast_signal(date)
            for k, [m, v] in enumerate(zip(f_m, f_v)):
                self.forecast_mean[k][idx:idx + sig.p] = m
                self.forecast_var[k][idx:idx + sig.p, idx:idx + sig.p] = v
            idx += sig.p

        return self.forecast_mean, self.forecast_var

    def copy(self):

        newsignals = []
        for sig in self.signals:

            newsignals.append(sig.copy())

        return multisignal(newsignals)

    def add_signal(self, signal):
        """
        :param signal: A new signal to be added to the multisignal
        :return:
        """
        # Append the new signal on
        self.signals.append(signal)

        self.nsignals = len(self.signals)
        self.p = np.sum([sig.p for sig in self.signals])
        self.k = np.min([sig.k for sig in self.signals])

        # initialize matrices that filled in when 'get_signal' and 'get_forecast_signal' are called
        self.mean = np.zeros(self.p)
        self.var = np.zeros([self.p, self.p])
        self.forecast_mean = [np.zeros(self.p) for k in range(self.k)]
        self.forecast_var = [np.zeros([self.p, self.p]) for k in range(self.k)]
        self.forecast_covar = [np.zeros([self.p, self.p]) for k in range(self.k)]

        # Set the start and end dates
        start_date = np.max([sig.start_date for sig in self.signals])
        end_date = np.min([sig.end_date for sig in self.signals])
        self.dates = pd.date_range(start_date, end_date)

        # Set the start and end forecast dates
        forecast_start_date = np.max([sig.forecast_start_date for sig in self.signals])
        forecast_end_date = np.min([sig.forecast_end_date for sig in self.signals])
        self.forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)


#### A number of common signal generating functions
def hol_fxn(date, mod, X, **kwargs):
    is_hol = np.any(X[-mod.nhol:] != 0)
    mean = np.zeros(mod.nhol)
    var = np.zeros([mod.nhol, mod.nhol])
    if is_hol:
        idx = np.where(X[-mod.nhol:] != 0)[0][0]
        mean[idx] = X[-mod.nhol:] @ mod.m[mod.ihol]
        var[idx, idx] = X[-mod.nhol:] @ mod.C[np.ix_(mod.ihol, mod.ihol)] @ X[-mod.nhol:]

    return mean, var

def hol_forecast_fxn(date, mod, X, k, horizons, **kwargs):
    future_holiday_eff = list(map(lambda X, k: forecast_holiday_effect(mod, X, k),
                                  X[:, -mod.nhol:], horizons))
    hol_mean = [np.zeros(mod.nhol) for h in range(k)]
    hol_var = [np.zeros([mod.nhol, mod.nhol]) for h in range(k)]
    for h in range(k):
        if future_holiday_eff[h][0] != 0:
            idx = np.where(X[h, -mod.nhol:] != 0)[0][0]
            hol_mean[h][idx] = future_holiday_eff[h][0]
            hol_var[h][idx, idx] = future_holiday_eff[h][1]

    return hol_mean, hol_var

hol_signal = signal(gen_fxn = hol_fxn, gen_forecast_fxn=hol_forecast_fxn)

def Y_fxn(date, mod, Y, **kwargs):
    return Y, 0

def Y_forecast_fxn(date, mod, X, k, nsamps, horizons, **kwargs):
    forecast = list(map(lambda X, k: mod.forecast_marginal(k=k, X=X, nsamps=nsamps),
                        X,
                        horizons))
    Y_mean = [f.mean() for f in forecast]
    Y_var = [f.var() for f in forecast]
    return Y_mean, Y_var

Y_signal = signal(gen_fxn = Y_fxn, gen_forecast_fxn = Y_forecast_fxn)

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


def seas_weekly_forecast_fxn(date, mod, k, horizons, **kwargs):
    period = 7
    future_weekly_seas = list(map(lambda k: forecast_weekly_seasonal_factor(mod, k=k),
                                  horizons))

    # Place the weekly seasonal factor into the correct spot in a length 7 vector
    today = date.dayofweek
    weekly_seas_mean = [np.zeros(period) for i in range(k)]
    weekly_seas_var = [np.zeros([period, period]) for i in range(k)]
    for i in range(k):
        day = (today + i) % period
        weekly_seas_mean[i][day] = future_weekly_seas[i][0]
        weekly_seas_var[i][day, day] = future_weekly_seas[i][1]

    return weekly_seas_mean, weekly_seas_var

seas_weekly_signal = signal(gen_fxn = seas_weekly_fxn, gen_forecast_fxn=seas_weekly_forecast_fxn)

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

pois_coef_signal = signal(gen_fxn = pois_coef_fxn, gen_forecast_fxn=pois_coef_forecast_fxn)

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

bern_coef_signal = signal(gen_fxn=bern_coef_fxn, gen_forecast_fxn=bern_coef_forecast_fxn)


def copy_fxn(date, signal):
    s = signal.get_signal(date)
    return copy.deepcopy(s[0]), copy.deepcopy(s[1])

def copy_forecast_fxn(date, signal):
    means = []
    vars = []
    ms, vs = signal.get_forecast_signal(date)
    for h in range(signal.k):
        means.append(copy.deepcopy(ms[h]))
        vars.append(copy.deepcopy(vs[h]))
    return means, vars

def merge_fxn(date, signals, **kwargs):
    if signals[0].p == 1:
        m = np.array([float(sig.get_signal(date)[0]) for sig in signals])
        v = np.array([float(sig.get_signal(date)[1]) for sig in signals])
        p = 1 / v
        return np.sum(m * p) / np.sum(p), 1 / np.sum(p)
    else:
        ms = [sig.get_signal(date)[0] for sig in signals]
        vs = [sig.get_signal(date)[1] for sig in signals]
        ps = [np.linalg.inv(v) for v in vs]
        m = np.sum([p @ m.reshape(-1,1) for m, p in zip(ms, ps)], axis=0)
        v = np.linalg.inv(np.sum(ps, axis=0))
        mean = v @ m
        return mean.reshape(-1), v

def merge_forecast_fxn(date, signals, **kwargs):
    k = np.min([sig.k for sig in signals])
    signal_mean = []
    signal_var = []
    if signals[0].p == 1:
        ms, vs = list(zip(*[sig.get_forecast_signal(date) for sig in signals]))
        for h in range(k):
            m = np.array([float(m[h]) for m in ms])
            v = np.array([float(v[h]) for v in vs])
            p = 1 / v
            signal_mean.append(np.sum(m * p) / np.sum(p))
            signal_var.append(1 / np.sum(p))
        return signal_mean, signal_var
    else:
        ms, vs = list(zip(*[sig.get_forecast_signal(date) for sig in signals]))
        ps = [[np.linalg.inv(var) for var in v] for v in vs]
        for h in range(k):
            m = np.sum([p[h] @ m[h].reshape(-1, 1) for m, p in zip(ms, ps)], axis=0)
            v = np.linalg.inv(np.sum([p[h] for p in ps], axis=0))
            mean = v @ m
            signal_mean.append(mean.reshape(-1))
            signal_var.append(v)
        return signal_mean, signal_var

def merge_signals(signals):
    """
    :param signals: list of the same signal from different sources to be combined into 1 using precision weighted averaging
    :return: A single signal
    """
    # Set the start and end dates
    start_date = np.min([sig.start_date for sig in signals])
    end_date = np.max([sig.end_date for sig in signals])
    dates = pd.date_range(start_date, end_date)

    # Set the start and end forecast dates
    forecast_start_date = np.min([sig.forecast_start_date for sig in signals])
    forecast_end_date = np.max([sig.forecast_end_date for sig in signals])
    forecast_dates = pd.date_range(forecast_start_date, forecast_end_date)

    # Create a new signal
    merged_sig = signal(gen_fxn = merge_fxn,
                    gen_forecast_fxn = merge_forecast_fxn)

    for date in dates:
        merged_sig.generate_signal(date, signals=[sig for sig in signals if sig.dates.isin([date]).any()])

    for date in forecast_dates:
        merged_sig.generate_forecast_signal(date, signals=[sig for sig in signals if sig.forecast_dates.isin([date]).any()])

    merged_sig.append_signal()
    merged_sig.append_forecast_signal()

    return merged_sig

def merge_sig_with_predictor(sig, X, X_dates):
    """
    Function to modify an uncertain signal by multiplying it by a known predictor. Example of uncertain signal is
     the coefficient on effect of price from an external model, while the price itself is a known predictor.

    :param X: A known predictor
    :param X_dates: Dates associated with the known predictor
    :return:
    """

    newsig = sig.copy()

    X = pd.DataFrame(X, index=X_dates)
    if sig.p == 1:

        for date in newsig.dates:
            if X_dates.isin([date]).any():
                newsig.mean.loc[date] *= X.loc[date].values
                newsig.var.loc[date] *= (X.loc[date].values ** 2)
            else:
                newsig.mean.drop(date, inplace=True)
                newsig.var.drop(date, inplace=True)

        for date in newsig.forecast_dates:
            if X_dates.isin([date]).any():

                m = newsig.forecast_mean.loc[date]
                v = newsig.forecast_var.loc[date]
                for h in range(newsig.k):
                    m[h] *= X.loc[date + pd.DateOffset(days=h)].values
                    v[h] *= (X.loc[date + pd.DateOffset(days=h)].values ** 2)
                newsig.forecast_mean.loc[date] = m
                newsig.forecast_var.loc[date] = v
            else:
                newsig.forecast_mean.drop(date, inplace=True)
                newsig.forecast_var.drop(date, inplace=True)


    else:
        for date in newsig.dates:
            if X_dates.isin([date]).any():
                newsig.mean.loc[date] *= X.loc[date].values
                newsig.var.loc[date] *= X.loc[date].values.reshape(-1,1) @ X.loc[date].values.reshape(1,-1)
            else:
                newsig.mean.drop(date, inplace=True)
                newsig.var.drop(date, inplace=True)

        for date in newsig.forecast_dates:
            if X_dates.isin([date]).any():
                m = newsig.forecast_mean.loc[date]
                v = newsig.forecast_var.loc[date]
                for h in range(newsig.k):
                    m[h] *= X.loc[date + pd.DateOffset(days=h)].values
                    v[h] *= X.loc[date + pd.DateOffset(days=h)].values.reshape(-1,1) @ X.loc[date + pd.DateOffset(days=h)].values.reshape(1,-1)
                newsig.forecast_mean.loc[date] = m
                newsig.forecast_var.loc[date] = v
            else:
                newsig.forecast_mean.drop(date, inplace=True)
                newsig.forecast_var.drop(date, inplace=True)

    return newsig

