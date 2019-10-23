import numpy as np
import pandas as pd

from .define_models import define_dglm
from .shared import define_holiday_regressors
from collections.abc import Iterable


def analysis(Y, X, k, forecast_start, forecast_end,
             nsamps=500, family = 'normal', n = None,
             model_prior = None, prior_length=20, ntrend=1,
             dates = None, holidays = [],
             seasPeriods = [], seasHarmComponents = [],
             ret=['model', 'forecast'], mean_only = False,
             **kwargs):
    """
    :param Y: Array of observations (n * 1)
    :param X: Array of covariates (n * p)
    :param k: forecast horizon (how many days ahead to forecast)
    :param forecast_start: date or index value to start forecasting (beginning with 0)
    :param forecast_end: date or index value to end forecasting
    :param family: Exponential family for Y. Options: 'normal', 'poisson', 'bernoulli', or 'binomial'
    :param nsamps: Number of forecast samples to draw
    :param n: If family is 'binomial', this is a (n * 1) array of test size (where Y is the array of successes)
    :param model_prior: A prespecified model of class DGLM
    :param prior_length: If model_prior is not given, a DGLM will be defined using the first 'prior_length' observations in Y, X.
    :param ntrend: Number of trend components in the model. 1 = local intercept , 2 = local intercept + local level
    :param dates: Array of dates (n * 1)
    :param holidays: List of Holidays to be given a special indicator (from pandas.tseries.holiday)
    :param seasPeriods: A list of periods for seasonal effects (e.g. [7] for a weekly effect, where Y is daily data)
    :param seasHarmComponents: A list of lists of harmonic components for a seasonal period (e.g. [[1,2,3]] if seasPeriods=[7])
    :param ret: A list of values to return. Options include: ['model', 'forecast, 'model_coef']
    :param mean_only: True/False - return the mean only when forecasting, instead of samples?
    :param kwargs: Further key word arguments to define the model prior. Common arguments include discount factors deltrend, delregn, delseas, and delhol.
    :return:
    """

    # Add the holiday indicator variables to the regression matrix
    nhol = len(holidays)
    if nhol > 0:
        X = define_holiday_regressors(X, dates, holidays)

    if model_prior is None:
        mod = define_dglm(Y, X, family=family, n=n, prior_length=prior_length, ntrend=ntrend, nhol=nhol,
                                 seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                                 **kwargs)
    else:
        mod = model_prior


    # Convert dates into row numbers
    if dates is not None:
        dates = pd.Series(dates)
        if type(forecast_start) == type(dates.iloc[0]):
            forecast_start = np.where(dates == forecast_start)[0][0]
        if type(forecast_end) == type(dates.iloc[0]):
            forecast_end = np.where(dates == forecast_end)[0][0]

    # Define the run length
    T = np.min([len(Y), forecast_end]) + 1

    if ret.__contains__('model_coef'):
        m = np.zeros([T, mod.a.shape[0]])
        C = np.zeros([T, mod.a.shape[0], mod.a.shape[0]])
        if family == 'normal':
            n = np.zeros(T)
            s = np.zeros(T)

    # Create dummy variable if there are no regression covariates
    if X is None:
        X = np.array([None]*T).reshape(-1,1)
    else:
        if len(X.shape) == 1:
            X = X.reshape(-1,1)

    # Initialize updating + forecasting
    horizons = np.arange(1, k + 1)

    if mean_only:
        forecast = np.zeros([1, forecast_end - forecast_start + 1, k])
    else:
        forecast = np.zeros([nsamps, forecast_end - forecast_start + 1, k])

    for t in range(prior_length, T):

        if forecast_start <= t <= forecast_end:
            if t == forecast_start:
                print('beginning forecasting')

            if ret.__contains__('forecast'):
                if family == "binomial":
                    forecast[:, t - forecast_start, :] = np.array(list(map(
                        lambda k, n, x:
                        mod.forecast_marginal(k=k, n=n, X=x, nsamps=nsamps, mean_only=mean_only),
                        horizons, n[t + horizons - 1], X[t + horizons - 1, :]))).squeeze().T.reshape(-1, k)  # .reshape(-1, 1)
                else:
                    # Get the forecast samples for all the items over the 1:k step ahead marginal forecast distributions
                    forecast[:, t - forecast_start, :] = np.array(list(map(
                        lambda k, x:
                        mod.forecast_marginal(k=k, X=x, nsamps=nsamps, mean_only=mean_only),
                        horizons, X[t + horizons - 1, :]))).squeeze().T.reshape(-1, k)#.reshape(-1, 1)


        # Now observe the true y value, and update:
        if t < len(Y):
            if family == "binomial":
                mod.update(y=Y[t], X=X[t], n=n[t])
            else:
                mod.update(y=Y[t], X=X[t])

            if ret.__contains__('model_coef'):
                m[t,:] = mod.m.reshape(-1)
                C[t,:,:] = mod.C
                if family == 'normal':
                    n[t] = mod.n / mod.delVar
                    s[t] = mod.s

    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
        if obj == 'model_coef':
            mod_coef = {'m':m, 'C':C}
            if family == 'normal':
                mod_coef.update({'n':n, 's':s})

            out.append(mod_coef)

    if len(out) == 1:
        return out[0]
    else:
        return out

