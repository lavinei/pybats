import numpy as np
import pandas as pd
from .define_models import define_dcmm, define_normal_dlm, define_dbcm
from .multiscale import get_latent_factor, forecast_latent_factor, sample_latent_factor
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday


def analysis_dcmm(Y, X, prior_length, k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                  phi_mu_prior = None, phi_sigma_prior = None, phi_psi_prior = None,
                  phi_mu_post = None, phi_sigma_post = None, mean_only=False, dates=None,
                  holidays = [], seasPeriods = [], seasHarmComponents = [], **kwargs):
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


    if phi_mu_prior is not None:
        multiscale = True
        nmultiscale = len(phi_mu_post[0])
    else:
        multiscale = False
        nmultiscale = 0

    # Add the holiday indicator variables to the regression matrix
    if holidays is not None:
        X = define_holiday_regressors(X, dates, holidays)

    # Initialize the DCMM
    mod = define_dcmm(Y, X, prior_length = prior_length, seasPeriods = seasPeriods, seasHarmComponents = seasHarmComponents,
                      nmultiscale = nmultiscale, rho = rho, **kwargs)

    # Initialize updating + forecasting
    horizons = np.arange(1,k+1)

    # Convert dates into row numbers
    if dates is not None and type(forecast_start) == type(dates[0]):
        forecast_start = np.where(dates == forecast_start)[0][0]
    if dates is not None and type(forecast_end) == type(dates[0]):
        forecast_end = np.where(dates == forecast_end)[0][0]

    if mean_only:
        forecast = np.zeros([1, forecast_end - forecast_start, k])
    else:
        forecast = np.zeros([nsamps, forecast_end - forecast_start, k])

    T = np.min([len(Y), forecast_end])
    nu = 9

    # Run updating + forecasting
    for t in range(prior_length, T):
        if t % 100 == 0:
            print(t)

        if t >= forecast_start and t < forecast_end:

            # Get the forecast samples for all the items over the 1:14 step ahead path
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
                            k, (x, x), (pm, pm), (ps, ps), nsamps=nsamps, mean_only=mean_only),
                        horizons, X[t + horizons - 1, :], pm, ps))).reshape(1, -1)
                else:
                    forecast[:, t - forecast_start, :] = mod.multiscale_forecast_path_approx(
                    k, (X[t + horizons - 1, :], X[t + horizons - 1, :]),
                    (pm, pm), (ps, ps), (pp, pp), nsamps=nsamps, t_dist=True, nu=nu)
            else:
                if mean_only:
                    forecast[:, t - forecast_start, :] = np.array(list(map(
                        lambda k, x: mod.forecast_marginal(k, (x, x), nsamps=nsamps, mean_only=mean_only),
                        horizons, X[t + horizons - 1, :]))).reshape(1,-1)
                else:
                    forecast[:, t - forecast_start, :] = mod.forecast_path_approx(
                    k, (X[t + horizons - 1, :], X[t + horizons - 1, :]), nsamps=nsamps, t_dist=True, nu=nu)

        # Update the DCMM
        if multiscale:
            pm = phi_mu_post[t-prior_length]
            ps = phi_sigma_post[t-prior_length]
            mod.multiscale_update_approx(y=Y[t], X=(X[t], X[t]),
                                         phi_mu=(pm, pm), phi_sigma=(ps, ps))
        else:
            mod.update(y = Y[t], X=(X[t], X[t]))

    return forecast

def analysis_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade, excess,
                  prior_length, k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                  phi_mu_prior = None, phi_sigma_prior = None, phi_psi_prior = None,
                  phi_mu_post = None, phi_sigma_post = None, mean_only=False, dates=None,
                  holidays = [], seasPeriods = [], seasHarmComponents = [], **kwargs):
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

    if phi_mu_prior is not None:
        multiscale = True
        nmultiscale = len(phi_mu_post[0])
    else:
        multiscale = False
        nmultiscale = 0


    if kwargs.get('period') is not None:
        period = kwargs.get('period') # This line is used for a seasonal multiscale latent factor

    # Add the holiday indicator variables to the regression matrix
    if holidays is not None:
        X_transaction = define_holiday_regressors(X_transaction, dates, holidays)

    mod = define_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade,
                      excess_values = excess, prior_length = prior_length,
                      seasPeriods = seasPeriods, seasHarmComponents=seasHarmComponents,
                      nmultiscale = nmultiscale, rho = rho, **kwargs)

    # Initialize updating + forecasting
    horizons = np.arange(1,k+1)

    # Convert dates into row numbers
    if dates is not None and type(forecast_start) == type(dates[0]):
        forecast_start = np.where(dates == forecast_start)[0][0]
    if dates is not None and type(forecast_end) == type(dates[0]):
        forecast_end = np.where(dates == forecast_end)[0][0]

    if mean_only:
        forecast = np.zeros([1, forecast_end - forecast_start, k])
    else:
        forecast = np.zeros([nsamps, forecast_end - forecast_start, k])

    T = np.min([len(Y_transaction), forecast_end])
    nu = 9

    # Run updating + forecasting
    for t in range(prior_length, T):
        if t % 100 == 0:
            print(t)

        if t >= forecast_start and t < forecast_end:

            # Get the forecast samples for all the items over the 1:14 step ahead path
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
                            k, x_trans, x_cascade, pm, ps, nsamps=nsamps, mean_only=mean_only),
                        horizons, X_transaction[t + horizons - 1, :], X_cascade[t + horizons - 1, :], pm, ps))).reshape(1, -1)
                else:
                    forecast[:, t - forecast_start, :] = mod.multiscale_forecast_path_approx(
                        k, X_transaction[t + horizons - 1, :], X_cascade[t + horizons - 1, :],
                        pm, ps, pp, nsamps=nsamps, t_dist=True, nu=nu)
            else:
                if mean_only:
                    forecast[:, t - forecast_start, :] = np.array(list(map(
                        lambda k, x_trans, x_cascade: mod.forecast_marginal(k, x_trans, x_cascade, nsamps=nsamps, mean_only=mean_only),
                        horizons, X_transaction[t + horizons - 1, :], X_cascade[t + horizons - 1, :]))).reshape(1,-1)
                else:
                    forecast[:, t - forecast_start, :] = mod.forecast_path_approx(
                        k, X_transaction[t + horizons - 1, :], X_cascade[t + horizons - 1, :],
                        nsamps=nsamps, t_dist=True, nu=nu)

        # Update the DCMM
        if multiscale:
            pm = phi_mu_post[t-prior_length]
            ps = phi_sigma_post[t-prior_length]
            mod.multiscale_update_approx(Y_transaction[t], X_transaction[t, :], Y_cascade[t,:], X_cascade[t, :],
                                         pm, ps, excess[t])
        else:
            mod.update(Y_transaction[t], X_transaction[t, :], Y_cascade[t,:], X_cascade[t, :], excess[t])

    return forecast


def analysis_lognormal_seasonalms(Y, X, prior_length, k, forecast_start, forecast_end, period=7, dates=None, **kwargs):
    """
    :param Y: Array of daily sales (typically on the log scale) (n * 1)
    :param X: Array of covariates (n * p)
    :param prior_length: number of datapoints to use for prior specification
    :param k: forecast horizon (how many days ahead to forecast)
    :param forecast_start: day to start forecasting (beginning with 0)
    :param forecast_end: day to end forecasting
    :param period: Integer, giving periodicity of the seasonal component that we want to extract for multiscale inference
    :param kwargs: Extra arguments used to initialized the model
    :return:
    """
    nmod = define_normal_dlm(Y, prior_length, **kwargs)
    horizons = np.arange(1, k + 1)
    phi_mu_prior = []
    phi_sigma_prior = []
    phi_mu_post = []
    phi_sigma_post = []

    # Convert dates into row numbers
    if dates is not None and type(forecast_start) == type(dates[0]):
        forecast_start = np.where(dates == forecast_start)[0][0]
    if dates is not None and type(forecast_end) == type(dates[0]):
        forecast_end = np.where(dates == forecast_end)[0][0]

    T = np.min([len(Y), forecast_end])

    for t in range(prior_length, T):
        # Forecast the latent factors (for forecasting and updating)
        today = t % period

        if t % 100 == 0:
            print(t)

        if t >= forecast_start and t < forecast_end:
            # Forecast the latent factor
            future_latent_factors = list(map(lambda k: forecast_latent_factor(nmod, k=k, today=today, period=period),
                                             horizons))
            phi_mu = [lf[0] for lf in future_latent_factors]
            phi_sigma = [lf[1] for lf in future_latent_factors]

            phi_mu_prior.append(phi_mu)
            phi_sigma_prior.append(phi_sigma)

        # Now observe the true y value, and update:

        # Update the normal DLM for total sales
        nmod.update(y=Y[t], X=X[t])
        phi_mu, phi_sigma = get_latent_factor(nmod, day=today)
        phi_mu_post.append(phi_mu)
        phi_sigma_post.append(phi_sigma)

    return phi_mu_prior, phi_sigma_prior, phi_mu_post, phi_sigma_post


def define_holiday_regressors(X, dates, holidays=None):
    n = X.shape[0]
    for holiday in holidays:
        cal = AbstractHolidayCalendar()
        cal.rules = [holiday]
        x = np.zeros(n)
        x[dates.isin(cal.holidays())] = 1
        X = np.c_[X, x]

    return X