import numpy as np
import pandas as pd

from .multiscale import forecast_holiday_effect
from .define_models import define_dcmm, define_normal_dlm, define_dbcm, define_amhm
from .seasonal import get_seasonal_effect_fxnl, forecast_weekly_seasonal_factor
from .shared import define_holiday_regressors
from .forecast import forecast_aR
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from collections import Iterable


def analysis_dcmm(Y, X, prior_length, k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                  phi_mu_prior = None, phi_sigma_prior = None, phi_psi_prior = None,
                  phi_mu_post = None, phi_sigma_post = None, mean_only=False, dates=None,
                  holidays = [], seasPeriods = [], seasHarmComponents = [], ret=['forecast'], **kwargs):
    """
    # Run updating + forecasting using a dcmm. Multiscale option available
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

    if ret.__contains__('pois_coef'):
        pois_coef_mean_prior = []
        pois_coef_var_prior = []
        pois_coef_mean_post = []
        pois_coef_var_post = []


    if phi_mu_prior is not None:
        multiscale = True
        nmultiscale = len(phi_mu_post[0])
    else:
        multiscale = False
        nmultiscale = 0

    # Convert dates into row numbers
    if dates is not None:
        dates = pd.to_datetime(dates, format='%y/%m/%d')
        if type(forecast_start) == type(dates.iloc[0]):
            forecast_start = np.where(dates == forecast_start)[0][0]
        if type(forecast_end) == type(dates.iloc[0]):
            forecast_end = np.where(dates == forecast_end)[0][0]

    # Add the holiday indicator variables to the regression matrix
    nhol = len(holidays)
    if nhol > 0:
        X = define_holiday_regressors(X, dates, holidays)

    # Initialize the DCMM
    mod = define_dcmm(Y, X, prior_length = prior_length, seasPeriods = seasPeriods, seasHarmComponents = seasHarmComponents,
                      nmultiscale = nmultiscale, rho = rho, nhol = nhol, **kwargs)

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
                    pm = phi_mu_prior[t-forecast_start]
                    ps = phi_sigma_prior[t-forecast_start]
                    if phi_psi_prior is not None:
                        pp = phi_psi_prior[t-forecast_start]
                    else:
                        pp = None

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

        if ret.__contains__('pois_coef'):
            if t >= forecast_start and t <= forecast_end:
                # Forecast the coefficients k-steps ahead
                pois_coef_mean = []
                pois_coef_var = []
                for j in range(1, k + 1):
                    a, R = forecast_aR(mod.pois_mod, j)
                    pois_coef_mean.append(a.copy())
                    pois_coef_var.append(R.diagonal().copy())
                pois_coef_mean_prior.append(pois_coef_mean)
                pois_coef_var_prior.append(pois_coef_var)

        # Update the DCMM
        if multiscale:
            pm = phi_mu_post[t-prior_length]
            ps = phi_sigma_post[t-prior_length]
            mod.multiscale_update_approx(y=Y[t], X=(X[t], X[t]),
                                         phi_mu=(pm, pm), phi_sigma=(ps, ps))
        else:
            mod.update(y = Y[t], X=(X[t], X[t]))

        if ret.__contains__('pois_coef'):
            pois_coef_mean_post.append(mod.pois_mod.m.copy())
            pois_coef_var_post.append(mod.pois_mod.C.diagonal().copy())

    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
        if obj == 'pois_coef':
            out.append(pois_coef_mean_prior)
            out.append(pois_coef_var_prior)
            out.append(pois_coef_mean_post)
            out.append(pois_coef_var_post)
    return out

def analysis_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade, excess,
                  prior_length, k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                  phi_mu_prior = None, phi_sigma_prior = None, phi_psi_prior = None,
                  phi_mu_post = None, phi_sigma_post = None, mean_only=False, dates=None, amhm=False,
                  holidays = [], seasPeriods = [], seasHarmComponents = [], ret=['forecast'], ret_fxn = None, **kwargs):
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

    if ret.__contains__('pois_coef'):
        pois_coef_mean_prior = []
        pois_coef_var_prior = []
        pois_coef_mean_post = []
        pois_coef_var_post = []

    if ret.__contains__('bern_coef'):
        bern_coef_mean_prior = []
        bern_coef_var_prior = []
        bern_coef_mean_post = []
        bern_coef_var_post = []

    if ret.__contains__('custom'):
        # Then we must have defined a return function
        if ret_fxn is None:
            print('You must define the custom return function')
        custom_mean_prior = []
        custom_var_prior = []
        custom_mean_post = []
        custom_var_post = []

    if phi_mu_prior is not None:
        multiscale = True
        nmultiscale = len(phi_mu_post[0])
    else:
        multiscale = False
        nmultiscale = 0

    # Convert dates into row numbers
    if dates is not None:
        dates = pd.to_datetime(dates, format='%y/%m/%d')
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

    if amhm:
        mod = define_amhm(mod, dates, holidays, prior_length=prior_length)

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
                    pm = phi_mu_prior[t-forecast_start]
                    ps = phi_sigma_prior[t-forecast_start]
                    if phi_psi_prior is not None:
                        pp = phi_psi_prior[t-forecast_start]
                    else:
                        pp = None

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

        if ret.__contains__('pois_coef'):
            if t >= forecast_start and t <= forecast_end:
                # Forecast the coefficients k-steps ahead
                pois_coef_mean = []
                pois_coef_var = []
                for j in range(1, k+1):
                    a, R = forecast_aR(mod.dcmm.pois_mod, j)
                    pois_coef_mean.append(a.copy())
                    pois_coef_var.append(R.diagonal().copy())
                pois_coef_mean_prior.append(pois_coef_mean)
                pois_coef_var_prior.append(pois_coef_var)

        if ret.__contains__('bern_coef'):
            if t >= forecast_start and t <= forecast_end:
                # Forecast the coefficients k-steps ahead
                bern_coef_mean = []
                bern_coef_var = []
                for j in range(1, k + 1):
                    a, R = forecast_aR(mod.dcmm.bern_mod, j)
                    bern_coef_mean.append(a.copy())
                    bern_coef_var.append(R.diagonal().copy())
                bern_coef_mean_prior.append(bern_coef_mean)
                bern_coef_var_prior.append(bern_coef_var)

        if ret.__contains__('custom'):
            if t >= forecast_start and t <= forecast_end:
                custom_mean_prior.append(ret_fxn(mod, type = 'mean_prior'))
                custom_var_prior.append(ret_fxn(mod, type='var_prior'))


        # Update the DBCM
        if multiscale:
            pm = phi_mu_post[t-prior_length]
            ps = phi_sigma_post[t-prior_length]
            mod.multiscale_update_approx(y_transaction=Y_transaction[t], X_transaction= X_transaction[t, :],
                                         y_cascade=Y_cascade[t,:], X_cascade=X_cascade[t, :],
                                         phi_mu=pm, phi_sigma=ps, excess=excess[t])
        else:
            mod.update(y_transaction=Y_transaction[t], X_transaction=X_transaction[t, :],
                       y_cascade=Y_cascade[t,:], X_cascade=X_cascade[t, :], excess=excess[t])

        if ret.__contains__('pois_coef'):
            pois_coef_mean_post.append(mod.dcmm.pois_mod.m.copy())
            pois_coef_var_post.append(mod.dcmm.pois_mod.C.diagonal().copy())

        if ret.__contains__('bern_coef'):
            bern_coef_mean_post.append(mod.dcmm.bern_mod.m.copy())
            bern_coef_var_post.append(mod.dcmm.bern_mod.C.diagonal().copy())

        if ret.__contains__('custom'):
            custom_mean_post.append(ret_fxn(mod, type='mean_post'))
            custom_var_post.append(ret_fxn(mod, type='var_post'))


    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
        if obj == 'pois_coef':
            out.append(pois_coef_mean_prior)
            out.append(pois_coef_var_prior)
            out.append(pois_coef_mean_post)
            out.append(pois_coef_var_post)
        if obj == 'bern_coef':
            out.append(bern_coef_mean_prior)
            out.append(bern_coef_var_prior)
            out.append(bern_coef_mean_post)
            out.append(bern_coef_var_post)
        if obj == 'custom':
            out.append(custom_mean_prior)
            out.append(custom_var_prior)
            out.append(custom_mean_post)
            out.append(custom_var_post)

    return out

def analysis_dbcm_new(Y_transaction, X_transaction, Y_cascade, X_cascade, excess,
                      prior_length, k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                      multiscale_signal = None,
                      mean_only=False, dates=None, amhm=False,
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

    def get_X(X, t): return X[t]
    if isinstance(X_transaction, Iterable):
        if len(X_transaction) == 2:
            def get_X(X, t): return (X[0][t], X[1][t])

    if multiscale_signal is not None:
        multiscale = True
        nmultiscale = multiscale_signal.p
    else:
        multiscale = False
        nmultiscale = 0

    # Convert dates into row numbers
    if dates is not None:
        dates = pd.to_datetime(dates, format='%y/%m/%d')
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

    if amhm:
        mod = define_amhm(mod, dates, holidays, prior_length=prior_length)

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
            out.append(new_signals)

    return out


def analysis_dlm(Y, X, prior_length, k, forecast_start, forecast_end, nsamps=500, holidays = [],
                 seasPeriods = [7], seasHarmComponents = [[1,2,3]],
                 phi_mu_prior = None, phi_sigma_prior = None, phi_psi_prior = None,
                 phi_mu_post = None, phi_sigma_post = None,
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

    # Check if multiscale DLM
    if phi_mu_prior is not None:
        multiscale = True
        nmultiscale = len(phi_mu_post[0])
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


    # Convert dates into row numbers
    if dates is not None:
        dates = pd.to_datetime(dates, format='%y/%m/%d')
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
                    pm = phi_mu_prior[t - forecast_start]
                    ps = phi_sigma_prior[t - forecast_start]
                    if phi_psi_prior is not None:
                        pp = phi_psi_prior[t - forecast_start]
                    else:
                        pp = None

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
            pm = phi_mu_post[t - prior_length]
            ps = phi_sigma_post[t - prior_length]
            nmod.multiscale_update_approx(y=Y[t], X=X[t],
                                          phi_mu=pm, phi_sigma=ps)
        else:
            nmod.update(y=Y[t], X=X[t])


        if ret.__contains__('new_signals'):
            for signal in new_signals:
                signal.generate_signal(date=dates[t], mod=nmod, Y=Y[t], X=X[t], k=k)

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
            out.append(new_signals)

        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(nmod)

    return out

