import numpy as np
import pandas as pd

from .multiscale import forecast_holiday_effect
from .define_models import define_dcmm, define_normal_dlm, define_dbcm
from .seasonal import get_seasonal_effect_fxnl, forecast_weekly_seasonal_factor
from .shared import define_holiday_regressors
from .forecast import forecast_aR
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from collections.abc import Iterable


def analysis_dcmm(Y, X, prior_length, k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                  multiscale_signal = None, mean_only=False, dates=None,
                  holidays = [], seasPeriods = [], seasHarmComponents = [], ret=['forecast'],
                  new_signals = None, **kwargs):
    """
    Run updating + forecasting using a dcmm. Multiscale option available

    :param Y: Array of daily sales (n * 1)
    :param X: Covariate array (n * p)
    :param prior_length: number of datapoints to use for prior specification
    :param k: forecast horizon (how many days ahead to forecast)
    :param forecast_start: day to start forecasting (beginning with 0)
    :param forecast_end:  day to end forecasting
    :param nsamps: Number of forecast samples to draw
    :param rho: Random effect extension to the Poisson DGLM, handles overdispersion
    :param phi_mu_prior: Mean of latent factors over k-step horizon (if using a multiscale DCMM)
    :param phi_sigma_prior: Variance of latent factors over k-step horizon (if using a multiscale DCMM)
    :param phi_psi_prior: Covariance of latent factors over k-step horizon (if using a multiscale DCMM)
    :param phi_mu_post: Daily mean of latent factors for updating (if using a multiscale DCMM)
    :param phi_sigma_post: Daily variance of latent factors for updating (if using a multiscale DCMM)
    :param holidays: List of holiday dates
    :param kwargs: Other keyword arguments for initializing the model. e.g. delregn = [.99, .98] discount factors.
    :return: Array of forecasting samples, dimension (nsamps * (forecast_end - forecast_start) * k)
    """

    if multiscale_signal is not None:
        multiscale = True
        # Note: This assumes that the bernoulli & poisson components have the same number of multiscale components
        if isinstance(multiscale_signal, (list, tuple)):
            nmultiscale = multiscale_signal[0].p
        else:
            nmultiscale = multiscale_signal.p
    else:
        multiscale = False
        nmultiscale = 0

    # Convert dates into row numbers
    if dates is not None:
        dates = pd.Series(dates)
        # dates = pd.to_datetime(dates, format='%y/%m/%d')
        if type(forecast_start) == type(dates.iloc[0]):
            forecast_start = np.where(dates == forecast_start)[0][0]
        if type(forecast_end) == type(dates.iloc[0]):
            forecast_end = np.where(dates == forecast_end)[0][0]

    # Add the holiday indicator variables to the regression matrix
    nhol = len(holidays)
    if nhol > 0:
        X_transaction = define_holiday_regressors(X, dates, holidays)

    # Initialize the DCMM
    mod = define_dcmm(Y, X, prior_length = prior_length, seasPeriods = seasPeriods, seasHarmComponents = seasHarmComponents,
                      nmultiscale = nmultiscale, rho = rho, nhol = nhol, **kwargs)

    if ret.__contains__('new_signals'):
        if not isinstance(new_signals, Iterable):
            new_signals = [new_signals]

        tmp = []
        for sig in new_signals:
            tmp.append(sig.copy())
        new_signals = tmp

    # Initialize updating + forecasting
    horizons = np.arange(1,k+1)

    if mean_only:
        forecast = np.zeros([1, forecast_end - forecast_start + 1, k])
    else:
        forecast = np.zeros([nsamps, forecast_end - forecast_start + 1, k])

    T = np.min([len(Y), forecast_end]) + 1
    nu = 9

    # Run updating + forecasting
    for t in range(prior_length, T):
        # if t % 100 == 0:
        #     print(t)
        if ret.__contains__('forecast'):
            if t >= forecast_start and t <= forecast_end:
                if t == forecast_start:
                    print('beginning forecasting')

                # Get the forecast samples for all the items over the 1:k step ahead path
                if multiscale:
                    if isinstance(multiscale_signal, (list, tuple)):
                        pm_bern, ps_bern = multiscale_signal[0].get_forecast_signal(dates.iloc[t])
                        pm_pois, ps_pois = multiscale_signal[1].get_forecast_signal(dates.iloc[t])
                        pm = (pm_bern, pm_pois)
                        ps = (ps_bern, ps_pois)
                    else:
                        pm, ps = multiscale_signal.get_forecast_signal(dates.iloc[t])

                    pp = None  # Not including the path dependency of the multiscale signal

                    if mean_only:
                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x, pm, ps: mod.multiscale_forecast_marginal_approx(
                                k=k, X=(x, x), phi_mu=(pm, pm), phi_sigma=(ps, ps), nsamps=nsamps, mean_only=mean_only),
                            horizons, X[t + horizons - 1, :], pm, ps))).reshape(1, -1)
                    else:
                        forecast[:, t - forecast_start, :] = mod.multiscale_forecast_path_approx(
                        k=k, X=(X[t + horizons - 1, :], X[t + horizons - 1, :]),
                        phi_mu=(pm, pm), phi_sigma=(ps, ps), phi_psi=(pp, pp), nsamps=nsamps, t_dist=True, nu=nu)
                else:
                    if mean_only:
                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x: mod.forecast_marginal(
                                k=k, X=(x, x), nsamps=nsamps, mean_only=mean_only),
                            horizons, X[t + horizons - 1, :]))).reshape(1,-1)
                    else:
                        forecast[:, t - forecast_start, :] = mod.forecast_path_approx(
                        k=k, X=(X[t + horizons - 1, :], X[t + horizons - 1, :]), nsamps=nsamps, t_dist=True, nu=nu)

        if ret.__contains__('new_signals'):
            if t >= forecast_start and t <= forecast_end:
                for signal in new_signals:
                    signal.generate_forecast_signal(date=dates.iloc[t], mod=mod, X=X[t + horizons - 1, :],
                                                    k=k, nsamps=nsamps, horizons=horizons)

        # Update the DCMM
        if multiscale:
            if isinstance(multiscale_signal, (list, tuple)):
                pm_bern, ps_bern = multiscale_signal[0].get_signal(dates.iloc[t])
                pm_pois, ps_pois = multiscale_signal[1].get_signal(dates.iloc[t])
                pm = (pm_bern, pm_pois)
                ps = (ps_bern, ps_pois)
            else:
                pm, ps = multiscale_signal.get_signal(dates.iloc[t])

            mod.multiscale_update_approx(y=Y[t], X=(X[t], X[t]),
                                         phi_mu=(pm, pm), phi_sigma=(ps, ps))
        else:
            mod.update(y = Y[t], X=(X[t], X[t]))

        if ret.__contains__('new_signals'):
            for signal in new_signals:
                signal.generate_signal(date=dates.iloc[t], mod=mod, X=X[t + horizons - 1, :],
                                       k=k, nsamps=nsamps, horizons=horizons)

    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
        if obj == 'new_signals':
            for signal in new_signals:
                signal.append_signal()
                signal.append_forecast_signal()
            if len(new_signals) == 1:
                out.append(new_signals[0])
            else:
                out.append(new_signals)

    if len(out) == 1:
        return out[0]
    else:
        return out

def analysis_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade, excess,
                      prior_length, k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                      multiscale_signal = None,
                      mean_only=False, dates=None,
                      holidays = [], seasPeriods = [], seasHarmComponents = [],
                      ret=['forecast'], new_signals = None, **kwargs):
    """
    # Run updating + forecasting using a dcmm. Multiscale option available
    :param Y_transaction: Array of daily transactions (n * 1)
    :param X_transaction: Covariate array (n * p)
    :param Y_cascade: Array of daily baskets of size r or greater, for 1 <= r <= ncascade
    :param X_cascade: Covariate array for the binomial DGLMs of the cascade
    :param prior_length: number of datapoints to use for prior specification
    :param k: forecast horizon (how many days ahead to forecast)
    :param forecast_start: day to start forecasting (beginning with 0)
    :param forecast_end:  day to end forecasting
    :param nsamps: Number of forecast samples to draw
    :param rho: Random effect extension to the Poisson DGLM, handles overdispersion
    :param phi_mu_prior: Mean of latent factors over k-step horizon (if using a multiscale DCMM)
    :param phi_sigma_prior: Variance of latent factors over k-step horizon (if using a multiscale DCMM)
    :param phi_psi_prior: Covariance of latent factors over k-step horizon (if using a multiscale DCMM)
    :param phi_mu_post: Daily mean of latent factors for updating (if using a multiscale DCMM)
    :param phi_sigma_post: Daily variance of latent factors for updating (if using a multiscale DCMM)
    :param kwargs: Other keyword arguments for initializing the model
    :return: Array of forecasting samples, dimension (nsamps * (forecast_end - forecast_start) * k)
    """

    if multiscale_signal is not None:
        multiscale = True
        # Note: This assumes that the bernoulli & poisson components have the same number of multiscale components
        if isinstance(multiscale_signal, (list, tuple)):
            nmultiscale = multiscale_signal[0].p
        else:
            nmultiscale = multiscale_signal.p
    else:
        multiscale = False
        nmultiscale = 0

    # Convert dates into row numbers
    if dates is not None:
        dates = pd.Series(dates)
        # dates = pd.to_datetime(dates, format='%y/%m/%d')
        if type(forecast_start) == type(dates.iloc[0]):
            forecast_start = np.where(dates == forecast_start)[0][0]
        if type(forecast_end) == type(dates.iloc[0]):
            forecast_end = np.where(dates == forecast_end)[0][0]

    # Add the holiday indicator variables to the regression matrix
    nhol = len(holidays)
    if nhol > 0:
        X_transaction = define_holiday_regressors(X_transaction, dates, holidays)


    mod = define_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade,
                      excess_values = excess, prior_length = prior_length,
                      seasPeriods = seasPeriods, seasHarmComponents=seasHarmComponents,
                      nmultiscale = nmultiscale, rho = rho, nhol=nhol, **kwargs)

    if ret.__contains__('new_signals'):
        if not isinstance(new_signals, Iterable):
            new_signals = [new_signals]

        tmp = []
        for sig in new_signals:
            tmp.append(sig.copy())
        new_signals = tmp

    # Initialize updating + forecasting
    horizons = np.arange(1,k+1)

    if mean_only:
        forecast = np.zeros([1, forecast_end - forecast_start + 1, k])
    else:
        forecast = np.zeros([nsamps, forecast_end - forecast_start + 1, k])

    T = np.min([len(Y_transaction)- k, forecast_end]) + 1
    nu = 9

    # Run updating + forecasting
    for t in range(prior_length, T):
        # if t % 100 == 0:
        #     print(t)
            # print(mod.dcmm.pois_mod.param1)
            # print(mod.dcmm.pois_mod.param2)
        if ret.__contains__('forecast'):
            if t >= forecast_start and t <= forecast_end:
                if t == forecast_start:
                    print('beginning forecasting')

                # Get the forecast samples for all the items over the 1:k step ahead path
                if multiscale:
                    if isinstance(multiscale_signal, (list, tuple)):
                        pm_bern, ps_bern = multiscale_signal[0].get_forecast_signal(dates.iloc[t])
                        pm_pois, ps_pois = multiscale_signal[1].get_forecast_signal(dates.iloc[t])
                        pm = (pm_bern, pm_pois)
                        ps = (ps_bern, ps_pois)
                    else:
                        pm, ps = multiscale_signal.get_forecast_signal(dates.iloc[t])

                    pp = None # Not including path dependency in multiscale signal

                    if mean_only:
                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x_trans, x_cascade, pm, ps: mod.multiscale_forecast_marginal_approx(
                                k=k, X_transaction=x_trans, X_cascade=x_cascade,
                                phi_mu=pm, phi_sigma=ps, nsamps=nsamps, mean_only=mean_only),
                            horizons, X_transaction[t + horizons - 1, :], X_cascade[t + horizons - 1, :], pm, ps))).reshape(1, -1)
                    else:
                        forecast[:, t - forecast_start, :] = mod.multiscale_forecast_path_approx(
                            k=k, X_transaction=X_transaction[t + horizons - 1, :], X_cascade=X_cascade[t + horizons - 1, :],
                            phi_mu=pm, phi_sigma=ps, phi_psi=pp, nsamps=nsamps, t_dist=True, nu=nu)
                else:
                    if mean_only:
                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x_trans, x_cascade: mod.forecast_marginal(
                                k=k, X_transaction=x_trans, X_cascade=x_cascade, nsamps=nsamps, mean_only=mean_only),
                            horizons, X_transaction[t + horizons - 1, :], X_cascade[t + horizons - 1, :]))).reshape(1,-1)
                    else:
                        forecast[:, t - forecast_start, :] = mod.forecast_path_approx(
                            k=k, X_transaction=X_transaction[t + horizons - 1, :], X_cascade=X_cascade[t + horizons - 1, :],
                            nsamps=nsamps, t_dist=True, nu=nu)

        if ret.__contains__('new_signals'):
            if t >= forecast_start and t <= forecast_end:
                for signal in new_signals:
                    signal.generate_forecast_signal(date=dates.iloc[t], mod=mod, X_transaction=X_transaction[t + horizons - 1, :],
                                                    X_cascade = X_cascade[t + horizons - 1, :],
                                                    k=k, nsamps=nsamps, horizons=horizons)
        # Update the DBCM
        if multiscale:
            if isinstance(multiscale_signal, (list, tuple)):
                pm_bern, ps_bern = multiscale_signal[0].get_signal(dates.iloc[t])
                pm_pois, ps_pois = multiscale_signal[1].get_signal(dates.iloc[t])
                pm = (pm_bern, pm_pois)
                ps = (ps_bern, ps_pois)
            else:
                pm, ps = multiscale_signal.get_signal(dates.iloc[t])

            mod.multiscale_update_approx(y_transaction=Y_transaction[t], X_transaction= X_transaction[t, :],
                                         y_cascade=Y_cascade[t,:], X_cascade=X_cascade[t, :],
                                         phi_mu=pm, phi_sigma=ps, excess=excess[t])
        else:
            mod.update(y_transaction=Y_transaction[t], X_transaction=X_transaction[t, :],
                       y_cascade=Y_cascade[t,:], X_cascade=X_cascade[t, :], excess=excess[t])

        if ret.__contains__('new_signals'):
            for signal in new_signals:
                signal.generate_signal(date=dates.iloc[t], mod=mod, X_transaction=X_transaction[t + horizons - 1, :],
                                       X_cascade = X_cascade[t + horizons - 1, :],
                                       k=k, nsamps=nsamps, horizons=horizons)

    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
        if obj == 'new_signals':
            for signal in new_signals:
                signal.append_signal()
                signal.append_forecast_signal()
            if len(new_signals) == 1:
                out.append(new_signals[0])
            else:
                out.append(new_signals)

    if len(out) == 1:
        return out[0]
    else:
        return out


def analysis_dlm(Y, X, prior_length, k, forecast_start, forecast_end, nsamps=500, holidays = [],
                 seasPeriods = [7], seasHarmComponents = [[1,2,3]],
                 multiscale_signal = None,
                 mean_only = False, dates=None, ret=['seasonal_weekly'], new_signals = None,
                 ntrend=2, **kwargs):
    """
    :param Y: Array of daily sales (typically on the log scale) (n * 1)
    :param X: Array of covariates (n * p)
    :param prior_length: number of datapoints to use for prior specification
    :param k: forecast horizon (how many days ahead to forecast)
    :param forecast_start: day to start forecasting (beginning with 0)
    :param forecast_end: day to end forecasting
    :param ret: Vector of items to return. Options include:
        'seasonal_weekly' - a vector of length 7, with day-of-week seasonal effects, and 6 zeros at all times
        'seasonal' - a single number capturing seasonal effects (less flexiblility for lower level models)
        'holidays' - a vector of length nhol, with the forecast holiday effects
    :param kwargs: Extra arguments used to initialized the model
    :return:
    """

    if multiscale_signal is not None:
        multiscale = True
        # Note: This assumes that the bernoulli & poisson components have the same number of multiscale components
        if isinstance(multiscale_signal, (list, tuple)):
            nmultiscale = multiscale_signal[0].p
        else:
            nmultiscale = multiscale_signal.p
    else:
        multiscale = False
        nmultiscale = 0

    # Add the holiday indicator variables to the regression matrix
    nhol = len(holidays)
    if nhol > 0:
        X = define_holiday_regressors(X, dates, holidays)

    nmod = define_normal_dlm(Y, X, prior_length, ntrend=ntrend, nhol=nhol, nmultiscale=nmultiscale,
                             seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                             **kwargs)

    if ret.__contains__('seasonal'):
        seas_mean_prior = []
        seas_var_prior = []
        seas_mean_post = []
        seas_var_post = []

    if ret.__contains__('mean_and_var'):
        mav_mean_prior = []
        mav_var_prior = []
        mav_mean_post = []
        mav_var_post = []

    if ret.__contains__('dof'):
        dof = []

    if ret.__contains__('new_signals'):
        if not isinstance(new_signals, Iterable):
            new_signals = [new_signals]

        tmp = []
        for sig in new_signals:
            tmp.append(sig.copy())
        new_signals = tmp

    # Convert dates into row numbers
    if dates is not None:
        dates = pd.Series(dates)
        # dates = pd.to_datetime(dates, format='%y/%m/%d')
        if type(forecast_start) == type(dates.iloc[0]):
            forecast_start = np.where(dates == forecast_start)[0][0]
        if type(forecast_end) == type(dates.iloc[0]):
            forecast_end = np.where(dates == forecast_end)[0][0]

    T = np.min([len(Y) - k, forecast_end]) + 1

    if X is None:
        X = np.array([None]*T).reshape(-1,1)

    # Initialize updating + forecasting
    horizons = np.arange(1, k + 1)

    if mean_only:
        forecast = np.zeros([1, forecast_end - forecast_start + 1, k])
    else:
        forecast = np.zeros([nsamps, forecast_end - forecast_start + 1, k])

    for t in range(prior_length, T):

        # if t % 100 == 0:
        #     print(t)

        if forecast_start <= t <= forecast_end:
            if t == forecast_start:
                print('beginning forecasting')

            if ret.__contains__('mean_and_var'):
                prior_mean = []
                prior_var = []
                for h in range(1, k+1):
                    mean, var = nmod.forecast_marginal(k=h, X=X[t + h - 1], state_mean_var=True)
                    # prior_mean.append(np.ravel(mean)[0])
                    # prior_var.append(np.ravel(var)[0])
                    prior_mean.append(mean)
                    prior_var.append(var)
                mav_mean_prior.append(prior_mean)
                mav_var_prior.append(prior_var)

            if ret.__contains__('new_signals'):
                for signal in new_signals:
                    signal.generate_forecast_signal(date=dates[t], mod=nmod, X=X[t + horizons - 1],
                                                    k=k, nsamps=nsamps, horizons=horizons)

            if ret.__contains__('forecast'):
                # Get the forecast samples for all the items over the 1:k step ahead path
                if multiscale:
                    pm, ps = multiscale_signal.get_forecast_signal(dates.iloc[t])
                    pp = None  # Not including path dependency in multiscale signal

                    forecast[:, t - forecast_start, :] = np.array(list(map(
                        lambda k, x, pm, ps: nmod.multiscale_forecast_marginal_approx(
                            k=k, X=x, phi_mu=pm, phi_sigma=ps, nsamps=nsamps, mean_only=mean_only),
                        horizons, X[t + horizons - 1, :], pm, ps))).reshape(-1,1)
                else:
                    forecast[:, t - forecast_start, :] = np.array(list(map(
                        lambda k, x: nmod.forecast_marginal(
                            k=k, X=x, nsamps=nsamps, mean_only=mean_only),
                        horizons, X[t + horizons - 1, :]))).reshape(-1, 1)


        # Now observe the true y value, and update:
        if ret.__contains__('mean_and_var'):
            mean, var = nmod.forecast_marginal(k=1, X = X[t], state_mean_var=True)
            # mav_mean_post.append(np.ravel(mean)[0])
            # mav_var_post.append(np.ravel(var)[0])
            mav_mean_post.append(mean)
            mav_var_post.append(var)

        if ret.__contains__('dof'):
            dof.append(nmod.n * nmod.delVar)

        # Update the normal DLM
        if multiscale:
            pm, ps = multiscale_signal.get_signal(dates.iloc[t])
            nmod.multiscale_update_approx(y=Y[t], X=X[t],
                                          phi_mu=pm, phi_sigma=ps)
        else:
            nmod.update(y=Y[t], X=X[t])


        if ret.__contains__('new_signals'):
            for signal in new_signals:
                signal.generate_signal(date=dates[t], mod=nmod, Y=Y[t], X=X[t], k=k, nsamps=nsamps)

    out = []
    for obj in ret:

        if obj == 'mean_and_var':
            out.append(mav_mean_prior)
            out.append(mav_var_prior)
            out.append(mav_mean_post)
            out.append(mav_var_post)

        if obj == 'dof':
            out.append(dof)

        if obj == 'new_signals':
            for signal in new_signals:
                signal.append_signal()
                signal.append_forecast_signal()
            if len(new_signals) == 1:
                out.append(new_signals[0])
            else:
                out.append(new_signals)

        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(nmod)

    if len(out) == 1:
        return out[0]
    else:
        return out

