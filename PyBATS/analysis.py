import numpy as np
import pandas as pd

from .multiscale import forecast_holiday_effect
from .define_models import define_dcmm, define_normal_dlm, define_dbcm, define_amhm
from .seasonal import get_seasonal_effect_fxnl, forecast_weekly_seasonal_factor
from .shared import define_holiday_regressors
from .forecast import forecast_aR
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday


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
    if ret.__contains__('forecast'): out.append(forecast)
    if ret.__contains__('model'): out.append(mod)
    if ret.__contains__('pois_coef'):
        out.append(pois_coef_mean_prior)
        out.append(pois_coef_var_prior)
        out.append(pois_coef_mean_post)
        out.append(pois_coef_var_post)
    return out

def analysis_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade, excess,
                  prior_length, k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                  phi_mu_prior = None, phi_sigma_prior = None, phi_psi_prior = None,
                  phi_mu_post = None, phi_sigma_post = None, mean_only=False, dates=None, amhm=False,
                  holidays = [], seasPeriods = [], seasHarmComponents = [], ret=['forecast'], **kwargs):
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

    out = []
    if ret.__contains__('forecast'): out.append(forecast)
    if ret.__contains__('model'): out.append(mod)
    if ret.__contains__('pois_coef'):
        out.append(pois_coef_mean_prior)
        out.append(pois_coef_var_prior)
        out.append(pois_coef_mean_post)
        out.append(pois_coef_var_post)
    return out


def analysis_dlm(Y, X, prior_length, k, forecast_start, forecast_end, nsamps=500, holidays = [],
                 seasPeriods = [7], seasHarmComponents = [[1,2,3]],
                 phi_mu_prior = None, phi_sigma_prior = None, phi_psi_prior = None,
                 phi_mu_post = None, phi_sigma_post = None,
                 mean_only = False, dates=None, ret=['seasonal_weekly'], ntrend=2, **kwargs):
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


    if ret.__contains__('seasonal_weekly'):
        period = 7
        seas_idx = np.where(np.array(nmod.seasPeriods) == 7)[0][0]
        weekly_seas_mean_prior = []
        weekly_seas_var_prior = []
        weekly_seas_mean_post = []
        weekly_seas_var_post = []

    if ret.__contains__('seasonal'):
        seas_mean_prior = []
        seas_var_prior = []
        seas_mean_post = []
        seas_var_post = []

    if ret.__contains__('holidays'):
        separate = True
        hol_mean_prior = []
        hol_var_prior = []
        hol_mean_post = []
        hol_var_post = []

    if ret.__contains__('Y'):
        Y_mean_prior = []
        Y_var_prior = []
        Y_mean_post = []
        Y_var_post = []

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

            # Forecast the weekly seasonal factor
            if ret.__contains__('seasonal_weekly'):
                future_weekly_seas = list(map(lambda k: forecast_weekly_seasonal_factor(nmod, k=k),
                                             horizons))

                # Place the weekly seasonal factor into the correct spot in a length 7 vector
                today = t % period
                weekly_seas_mean = [np.zeros(period) for i in range(k)]
                weekly_seas_var = [np.zeros([period, period]) for i in range(k)]
                for i in range(k):
                    day = (today + i) % period
                    weekly_seas_mean[i][day] = future_weekly_seas[i][0]
                    weekly_seas_var[i][day, day] = future_weekly_seas[i][1]
                weekly_seas_mean_prior.append(weekly_seas_mean)
                weekly_seas_var_prior.append(weekly_seas_var)

            # Forecast the future holiday effects
            if ret.__contains__('holidays'):
                future_holiday_eff = list(map(lambda X, k: forecast_holiday_effect(nmod, X, k),
                                              X[t + horizons - 1, -nmod.nhol:], horizons))
                if separate:
                    hol_mean = [np.zeros(nmod.nhol) for h in range(k)]
                    hol_var = [np.zeros([nmod.nhol, nmod.nhol]) for h in range(k)]
                    for h in range(k):
                        if future_holiday_eff[h][0] != 0:
                            idx = np.where(future_holiday_eff[h][0] != 0)[0][0]
                            hol_mean[h][idx] = future_holiday_eff[h][0]
                            hol_var[h][idx, idx] = future_holiday_eff[h][1]
                else:
                    hol_mean = [h[0] for h in future_holiday_eff]
                    hol_var = [h[1] for h in future_holiday_eff]


                hol_mean_prior.append(hol_mean)
                hol_var_prior.append(hol_var)

            # Forecast sales (could also do this analytically, currently using samples)
            if ret.__contains__('Y'):
                forecast = list(map(lambda X, k: nmod.forecast_marginal(k=k, X = X, nsamps=nsamps),
                                    X[t + horizons - 1],
                                    horizons))
                Y_mean = [f.mean() for f in forecast]
                Y_var = [f.var() for f in forecast]
                Y_mean_prior.append(Y_mean)
                Y_var_prior.append(Y_var)

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



        # Get the weekly seasonal factor for updating
        if ret.__contains__('seasonal_weekly'):
            today = t % period
            m, v = get_seasonal_effect_fxnl(nmod.L[seas_idx], nmod.m, nmod.C, nmod.iseas[seas_idx])
            weekly_seas_mean = np.zeros(period)
            weekly_seas_var = np.zeros([period, period])
            weekly_seas_mean[today] = m
            weekly_seas_var[today, today] = v
            weekly_seas_mean_post.append(weekly_seas_mean)
            weekly_seas_var_post.append(weekly_seas_var)

        # Get the holiday effects for updating
        if ret.__contains__('holidays'):
            if separate:
                is_hol = np.any(X[t, -nmod.nhol:] != 0)
                mean = np.zeros(nmod.nhol)
                var = np.zeros([nmod.nhol, nmod.nhol])
                if is_hol:
                    idx = np.where(X[t, -nmod.nhol:] != 0)[0][0]
                    mean[idx] = X[t, -nmod.nhol:] @ nmod.m[nmod.ihol]
                    var[idx, idx] = X[t, -nmod.nhol:] @ nmod.C[np.ix_(nmod.ihol, nmod.ihol)] @ X[t, -nmod.nhol:]
                hol_mean_post.append(mean)
                hol_var_post.append(var)
            else:
                hol_mean_post.append(X[t, -nmod.nhol:] @ nmod.m[nmod.ihol])
                hol_var_post.append(X[t, -nmod.nhol:] @ nmod.C[np.ix_(nmod.ihol, nmod.ihol)] @ X[t, -nmod.nhol:])

        # Get the sales
        if ret.__contains__('Y'):
            Y_mean_post.append(Y[t])
            Y_var_post.append(0)

    out = []
    if ret.__contains__('seasonal_weekly'):
        out.append(weekly_seas_mean_prior)
        out.append(weekly_seas_var_prior)
        out.append(weekly_seas_mean_post)
        out.append(weekly_seas_var_post)

    if ret.__contains__('holidays'):
        out.append(hol_mean_prior)
        out.append(hol_var_prior)
        out.append(hol_mean_post)
        out.append(hol_var_post)

    if ret.__contains__('Y'):
        out.append(Y_mean_prior)
        out.append(Y_var_prior)
        out.append(Y_mean_post)
        out.append(Y_var_post)

    if ret.__contains__('mean_and_var'):
        out.append(mav_mean_prior)
        out.append(mav_var_prior)
        out.append(mav_mean_post)
        out.append(mav_var_post)

    if ret.__contains__('dof'):
        out.append(dof)

    if ret.__contains__('forecast'): out.append(forecast)
    if ret.__contains__('model'): out.append(nmod)

    return out