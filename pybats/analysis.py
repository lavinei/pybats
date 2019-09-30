import numpy as np
import pandas as pd

from .define_models import define_dglm
from .shared import define_holiday_regressors
from collections.abc import Iterable


def analysis(Y, X,
             k, forecast_start, forecast_end, nsamps=500, mean_only = False,
             model_prior = None, family = 'normal', prior_length=20, ntrend=1,
             dates = None, holidays = [],
             seasPeriods = [], seasHarmComponents = [],
             latent_factor = None, new_latent_factors = None,
             ret=['model', 'forecast'],
             **kwargs):
    """
    :param Y: Array of observations (n * 1)
    :param X: Array of covariates (n * p)
    :param k: forecast horizon (how many days ahead to forecast)
    :param forecast_start: date or index value to start forecasting (beginning with 0)
    :param forecast_end: date or index value to end forecasting
    :param nsamps: Number of forecast samples to draw
    :param mean_only: True/False - return the mean only when forecasting, instead of samples?
    :param model_prior: A prespecified model of class DGLM
    :param family: Exponential family for Y. Options: 'normal', 'poisson', 'bernoulli'
    :param prior_length: If model_prior is not given, a DGLM will be defined using the first 'prior_length' observations in Y, X.
    :param ntrend: Number of trend components in the model. 1 = local intercept , 2 = local intercept + local level
    :param dates: Array of dates (n * 1)
    :param holidays: List of Holidays to be given a special indicator (from pandas.tseries.holiday)
    :param seasPeriods: A list of periods for seasonal effects (e.g. [7] for a weekly effect, where Y is daily data)
    :param seasHarmComponents: A list of lists of harmonic components for a seasonal period (e.g. [[1,2,3]] if seasPeriods=[7])
    :param latent_factor: A latent_factor or multi_latent_factor, to be used as a predictor
    :param new_latent_factors: A list of latent factors to be created from this model, and returned (automatically, doesn't need to be specified)
    :param ret: A list of values to return. Options include: ['model', 'forecast, 'model_coef']
    :param kwargs: Further key word arguments to define the model prior. Common arguments include discount factors deltrend, delregn, delseas, and delhol.
    :return:
    """

    # Add the holiday indicator variables to the regression matrix
    nhol = len(holidays)
    if nhol > 0:
        X = define_holiday_regressors(X, dates, holidays)

    if latent_factor is not None:
        is_lf = True
        nlf = latent_factor.p
    else:
        is_lf = False
        nlf = 0

    if model_prior is None:
        mod = define_dglm(Y, X, family=family, prior_length=prior_length, ntrend=ntrend, nhol=nhol, nlf=nlf,
                                 seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                                 **kwargs)
    else:
        mod = model_prior


    # Convert dates into row numbers
    if dates is not None:
        dates = pd.Series(dates)
        # dates = pd.to_datetime(dates, format='%y/%m/%d')
        if type(forecast_start) == type(dates.iloc[0]):
            forecast_start = np.where(dates == forecast_start)[0][0]
        if type(forecast_end) == type(dates.iloc[0]):
            forecast_end = np.where(dates == forecast_end)[0][0]

    # Define the run length
    T = np.min([len(Y) - k, forecast_end]) + 1

    if ret.__contains__('model_coef'):
        m = np.zeros(T, mod.p)
        C = np.zeros(T, mod.p, mod.p)
        if family == 'normal':
            n = np.zeros(T)
            s = np.zeros(T)

    if new_latent_factors is not None:
        if not ret.__contains__('new_latent_factors'):
            ret.append('new_latent_factors')

        if not isinstance(new_latent_factors, Iterable):
            new_latent_factors = [new_latent_factors]

        tmp = []
        for lf in new_latent_factors:
            tmp.append(lf.copy())
        new_latent_factors = tmp

    # Create dummy variable if there are no regression covariates
    if X is None:
        X = np.array([None]*T).reshape(-1,1)

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
                # Get the forecast samples for all the items over the 1:k step ahead marginal forecast distributions
                if is_lf:
                    pm, ps = latent_factor.get_lf_forecast(dates.iloc[t])
                    pp = None  # Not including path dependency in latent factor

                    forecast[:, t - forecast_start, :] = np.array(list(map(
                        lambda k, x, pm, ps:
                        mod.forecast_marginal_lf_analytic(k=k, X=x, phi_mu=pm, phi_sigma=ps, nsamps=nsamps, mean_only=mean_only),
                        horizons, X[t + horizons - 1, :], pm, ps))).squeeze().T.reshape(-1, k)#.reshape(-1, 1)
                else:
                    forecast[:, t - forecast_start, :] = np.array(list(map(
                        lambda k, x:
                        mod.forecast_marginal(k=k, X=x, nsamps=nsamps, mean_only=mean_only),
                        horizons, X[t + horizons - 1, :]))).squeeze().T.reshape(-1, k)#.reshape(-1, 1)

            if ret.__contains__('new_latent_factors'):
                for lf in new_latent_factors:
                    lf.generate_lf_forecast(date=dates[t], mod=mod, X=X[t + horizons - 1],
                                                k=k, nsamps=nsamps, horizons=horizons)

        # Now observe the true y value, and update:
        if is_lf:
            pm, ps = latent_factor.get_lf(dates.iloc[t])
            mod.update_lf_analytic(y=Y[t], X=X[t],
                                    phi_mu=pm, phi_sigma=ps)
        else:
            mod.update(y=Y[t], X=X[t])

        if ret.__contains__('model_coef'):
            m[t,:] = mod.m
            C[t,:,:] = mod.C
            if family == 'normal':
                n[t] = mod.n / mod.delVar
                s[t] = mod.s

        if ret.__contains__('new_latent_factors'):
            for lf in new_latent_factors:
                lf.generate_lf(date=dates[t], mod=mod, Y=Y[t], X=X[t], k=k, nsamps=nsamps)

    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
        if obj == 'model_coef':
            mod_coef = {'m':m, 'C':C}
            if family == 'normal':
                mod_coef.update({'n':n, 's':s})

            out.append(mod_coef)
        if obj == 'new_latent_factors':
            for lf in new_latent_factors:
                lf.append_lf()
                lf.append_lf_forecast()
            if len(new_latent_factors) == 1:
                out.append(new_latent_factors[0])
            else:
                out.append(new_latent_factors)

    if len(out) == 1:
        return out[0]
    else:
        return out

