import numpy as np
import pandas as pd

from .define_models import define_dglm, define_dcmm, define_dbcm
from .shared import define_holiday_regressors
from collections.abc import Iterable


def analysis(Y, X, k, forecast_start, forecast_end,
             nsamps=500, family = 'normal', n = None,
             model_prior = None, prior_length=20, ntrend=1,
             dates = None, holidays = [],
             seasPeriods = [], seasHarmComponents = [],
             latent_factor = None, new_latent_factors = None,
             ret=['model', 'forecast'],
             mean_only = False, forecast_path = False,
             **kwargs):
    """
    This is a helpful function to run a standard analysis. The function will:
    1. Automatically initialize a DGLM
    2. Run sequential updating
    3. Forecast at each specified time step

    This function has many arguments to specify the model components. Here's a simple example for a Poisson DGLM:

    .. code::

        mod, samples = analysis(Y, X, family="poisson",
                                forecast_start=15,      # First time step to forecast on
                                forecast_end=20,        # Final time step to forecast on
                                k=1,                    # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
                                prior_length=6,         # Number of observations to use in defining prior
                                rho=.5,                 # Random effect extension, increases variance of Poisson DGLM (see Berry and West, 2019)
                                deltrend=0.95,          # Discount factor on the trend component (intercept)
                                delregn=0.95            # Discount factor on the regression component
                                )


    :param Y: Array of observations (n * 1)
    :param X: Array of covariates (n * p)
    :param k: forecast horizon (how many days ahead to forecast)
    :param forecast_start: date or index value to start forecasting (beginning with 0).
    :param forecast_end: date or index value to end forecasting
    :param family: Exponential family for Y. Options: 'normal', 'poisson', 'bernoulli', or 'binomial'
    :param nsamps: Number of forecast samples to draw
    :param n: If family is 'binomial', this is a (n * 1) array of test size (where Y is the array of successes)
    :param model_prior: A prespecified model of class DGLM
    :param prior_length: If model_prior is not given, a DGLM will be defined using the first 'prior_length' observations in Y, X.
    :param ntrend: Number of trend components in the model. 1 = local intercept , 2 = local intercept + local level
    :param dates: Array of dates (n * 1). Must be specified if forecast_start and forecast_end are dates.
    :param holidays: List of Holidays to be given a special indicator (from pandas.tseries.holiday)
    :param seasPeriods: A list of periods for seasonal effects (e.g. [7] for a weekly effect, where Y is daily data)
    :param seasHarmComponents: A list of lists of harmonic components for a seasonal period (e.g. [[1,2,3]] if seasPeriods=[7])
    :param ret: A list of values to return. Options include: ['model', 'forecast, 'model_coef']
    :param mean_only: True/False - return the mean only when forecasting, instead of samples?
    :param kwargs: Further key word arguments to define the model prior. Common arguments include discount factors deltrend, delregn, delseas, and delhol.

    :return: Default is a list with [model, forecast_samples]. forecast_samples will have dimension (nsamps by forecast_length by k), where forecast_length is the number of time steps between forecast_start and forecast_end. The output can be customized with the 'ret' parameter, which is a list of objects to return.
    """

    # Add the holiday indicator variables to the regression matrix
    nhol = len(holidays)
    X = define_holiday_regressors(X, dates, holidays)

    # Check if it's a latent factor DGLM
    if latent_factor is not None:
        is_lf = True
        nlf = latent_factor.p
    else:
        is_lf = False
        nlf = 0

    if model_prior is None:
        mod = define_dglm(Y, X, family=family, n=n, prior_length=prior_length, ntrend=ntrend, nhol=nhol, nlf=nlf,
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
                if is_lf:
                    if forecast_path:
                        pm, ps, pp = latent_factor.get_lf_forecast(dates.iloc[t])
                        forecast[:, t - forecast_start, :] = mod.forecast_path_lf_copula(k=k, X=X[t + horizons - 1, :],
                                                                                         nsamps=nsamps,
                                                                                         phi_mu=pm, phi_sigma=ps, phi_psi=pp)
                    else:
                        pm, ps = latent_factor.get_lf_forecast(dates.iloc[t])
                        pp = None  # Not including path dependency in latent factor

                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x, pm, ps:
                            mod.forecast_marginal_lf_analytic(k=k, X=x, phi_mu=pm, phi_sigma=ps, nsamps=nsamps, mean_only=mean_only),
                            horizons, X[t + horizons - 1, :], pm, ps))).squeeze().T.reshape(-1, k)#.reshape(-1, 1)
                else:
                    if forecast_path:
                        forecast[:, t - forecast_start, :] = mod.forecast_path(k=k, X = X[t + horizons - 1, :], nsamps=nsamps)
                    else:
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

            if ret.__contains__('new_latent_factors'):
                for lf in new_latent_factors:
                    lf.generate_lf_forecast(date=dates[t], mod=mod, X=X[t + horizons - 1],
                                            k=k, nsamps=nsamps, horizons=horizons)

        # Now observe the true y value, and update:
        if t < len(Y):
            if is_lf:
                pm, ps = latent_factor.get_lf(dates.iloc[t])
                mod.update_lf_analytic(y=Y[t], X=X[t],
                                       phi_mu=pm, phi_sigma=ps)
            else:
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


def analysis_dcmm(Y, X, prior_length, k, forecast_start, forecast_end, nsamps=500, rho=.6, latent_factor=None,
                  mean_only=False, dates=None, holidays=[], seasPeriods=[], seasHarmComponents=[], ret=['forecast'],
                  new_latent_factors=None, **kwargs):
    """
    Run updating + forecasting using a dcmm. Latent Factor option available

    :param Y: Array of daily sales (n * 1)
    :param X: Covariate array (n * p)
    :param prior_length: number of datapoints to use for prior specification
    :param k: forecast horizon (how many days ahead to forecast)
    :param forecast_start: day to start forecasting (beginning with 0)
    :param forecast_end:  day to end forecasting
    :param nsamps: Number of forecast samples to draw
    :param rho: Random effect extension to the Poisson DGLM, handles overdispersion
    :param phi_mu_prior: Mean of latent factors over k-step horizon (if using a Latent Factor DCMM)
    :param phi_sigma_prior: Variance of latent factors over k-step horizon (if using a Latent Factor DCMM)
    :param phi_psi_prior: Covariance of latent factors over k-step horizon (if using a Latent Factor DCMM)
    :param phi_mu_post: Daily mean of latent factors for updating (if using a Latent Factor DCMM)
    :param phi_sigma_post: Daily variance of latent factors for updating (if using a Latent Factor DCMM)
    :param holidays: List of holiday dates
    :param kwargs: Other keyword arguments for initializing the model. e.g. delregn = [.99, .98] discount factors.
    :return: Array of forecasting samples, dimension (nsamps * (forecast_end - forecast_start) * k)
    """

    if latent_factor is not None:
        is_lf = True
        # Note: This assumes that the bernoulli & poisson components have the same number of latent factor components
        if isinstance(latent_factor, (list, tuple)):
            nlf = latent_factor[0].p
        else:
            nlf = latent_factor.p
    else:
        is_lf = False
        nlf = 0

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
    if not kwargs.__contains__('model_prior'):
        mod = define_dcmm(Y, X, prior_length = prior_length, seasPeriods = seasPeriods, seasHarmComponents = seasHarmComponents,
                          nlf = nlf, rho = rho, nhol = nhol, **kwargs)
    else:
        mod = kwargs.get('model_prior')

    if ret.__contains__('new_latent_factors'):
        if not isinstance(new_latent_factors, Iterable):
            new_latent_factors = [new_latent_factors]

        tmp = []
        for sig in new_latent_factors:
            tmp.append(sig.copy())
        new_latent_factors = tmp

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
                if is_lf:
                    if isinstance(latent_factor, (list, tuple)):
                        pm_bern, ps_bern = latent_factor[0].get_lf_forecast(dates.iloc[t])
                        pm_pois, ps_pois = latent_factor[1].get_lf_forecast(dates.iloc[t])
                        pm = (pm_bern, pm_pois)
                        ps = (ps_bern, ps_pois)
                    else:
                        pm, ps = latent_factor.get_lf_forecast(dates.iloc[t])

                    pp = None  # Not including the path dependency of the latent factor

                    if mean_only:
                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x, pm, ps: mod.forecast_marginal_lf_analytic(
                                k=k, X=(x, x), phi_mu=(pm, pm), phi_sigma=(ps, ps), nsamps=nsamps, mean_only=mean_only),
                            horizons, X[t + horizons - 1, :], pm, ps))).reshape(1, -1)
                    else:
                        forecast[:, t - forecast_start, :] = mod.forecast_path_lf_copula(
                        k=k, X=(X[t + horizons - 1, :], X[t + horizons - 1, :]),
                        phi_mu=(pm, pm), phi_sigma=(ps, ps), phi_psi=(pp, pp), nsamps=nsamps, t_dist=True, nu=nu)
                else:
                    if mean_only:
                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x: mod.forecast_marginal(
                                k=k, X=(x, x), nsamps=nsamps, mean_only=mean_only),
                            horizons, X[t + horizons - 1, :]))).reshape(1,-1)
                    else:
                        forecast[:, t - forecast_start, :] = mod.forecast_path_copula(
                        k=k, X=(X[t + horizons - 1, :], X[t + horizons - 1, :]), nsamps=nsamps, t_dist=True, nu=nu)

        if ret.__contains__('new_latent_factors'):
            if t >= forecast_start and t <= forecast_end:
                for lf in new_latent_factors:
                    lf.generate_lf_forecast(date=dates.iloc[t], mod=mod, X=X[t + horizons - 1, :],
                                                k=k, nsamps=nsamps, horizons=horizons)

        # Update the DCMM
        if is_lf:
            if isinstance(latent_factor, (list, tuple)):
                pm_bern, ps_bern = latent_factor[0].get_lf(dates.iloc[t])
                pm_pois, ps_pois = latent_factor[1].get_lf(dates.iloc[t])
                pm = (pm_bern, pm_pois)
                ps = (ps_bern, ps_pois)
            else:
                pm, ps = latent_factor.get_lf(dates.iloc[t])

            mod.update_lf_analytic(y=Y[t], X=(X[t], X[t]),
                                   phi_mu=(pm, pm), phi_sigma=(ps, ps))
        else:
            mod.update(y = Y[t], X=(X[t], X[t]))

        if ret.__contains__('new_latent_factors'):
            for lf in new_latent_factors:
                lf.generate_lf(date=dates.iloc[t], mod=mod, X=X[t + horizons - 1, :],
                                   k=k, nsamps=nsamps, horizons=horizons)

    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
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


def analysis_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade, excess,
                  prior_length, k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                  latent_factor = None,
                  mean_only=False, dates=None,
                  holidays = [], seasPeriods = [], seasHarmComponents = [],
                  ret=['forecast'], new_latent_factors = None, **kwargs):
    """
    # Run updating + forecasting using a dcmm. Latent Factor option available
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
    :param phi_mu_prior: Mean of latent factors over k-step horizon (if using a Latent Factor DCMM)
    :param phi_sigma_prior: Variance of latent factors over k-step horizon (if using a Latent Factor DCMM)
    :param phi_psi_prior: Covariance of latent factors over k-step horizon (if using a Latent Factor DCMM)
    :param phi_mu_post: Daily mean of latent factors for updating (if using a Latent Factor DCMM)
    :param phi_sigma_post: Daily variance of latent factors for updating (if using a Latent Factor DCMM)
    :param kwargs: Other keyword arguments for initializing the model
    :return: Array of forecasting samples, dimension (nsamps * (forecast_end - forecast_start) * k)
    """

    if latent_factor is not None:
        is_lf = True
        # Note: This assumes that the bernoulli & poisson components have the same number of latent factor components
        if isinstance(latent_factor, (list, tuple)):
            nlf = latent_factor[0].p
        else:
            nlf = latent_factor.p
    else:
        is_lf = False
        nlf = 0

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


    if not kwargs.__contains__('model_prior'):
        mod = define_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade,
                          excess_values = excess, prior_length = prior_length,
                          seasPeriods = seasPeriods, seasHarmComponents=seasHarmComponents,
                          nlf = nlf, rho = rho, nhol=nhol, **kwargs)
    else:
        mod = kwargs.get('model_prior')

    if ret.__contains__('new_latent_factors'):
        if not isinstance(new_latent_factors, Iterable):
            new_latent_factors = [new_latent_factors]

        tmp = []
        for sig in new_latent_factors:
            tmp.append(sig.copy())
        new_latent_factors = tmp

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
                if is_lf:
                    if isinstance(latent_factor, (list, tuple)):
                        pm_bern, ps_bern = latent_factor[0].get_lf_forecast(dates.iloc[t])
                        pm_pois, ps_pois = latent_factor[1].get_lf_forecast(dates.iloc[t])
                        pm = (pm_bern, pm_pois)
                        ps = (ps_bern, ps_pois)
                        pp = None  # Not including path dependency in latent factor
                    else:
                        if latent_factor.forecast_path:
                            pm, ps, pp = latent_factor.get_lf_forecast(dates.iloc[t])
                        else:
                            pm, ps = latent_factor.get_lf_forecast(dates.iloc[t])
                            pp = None

                    if mean_only:
                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x_trans, x_cascade, pm, ps: mod.forecast_marginal_lf_analytic(
                                k=k, X_transaction=x_trans, X_cascade=x_cascade,
                                phi_mu=pm, phi_sigma=ps, nsamps=nsamps, mean_only=mean_only),
                            horizons, X_transaction[t + horizons - 1, :], X_cascade[t + horizons - 1, :], pm, ps))).reshape(1, -1)
                    else:
                        forecast[:, t - forecast_start, :] = mod.forecast_path_lf_copula(
                            k=k, X_transaction=X_transaction[t + horizons - 1, :], X_cascade=X_cascade[t + horizons - 1, :],
                            phi_mu=pm, phi_sigma=ps, phi_psi=pp, nsamps=nsamps, t_dist=True, nu=nu)
                else:
                    if mean_only:
                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x_trans, x_cascade: mod.forecast_marginal(
                                k=k, X_transaction=x_trans, X_cascade=x_cascade, nsamps=nsamps, mean_only=mean_only),
                            horizons, X_transaction[t + horizons - 1, :], X_cascade[t + horizons - 1, :]))).reshape(1,-1)
                    else:
                        forecast[:, t - forecast_start, :] = mod.forecast_path_copula(
                            k=k, X_transaction=X_transaction[t + horizons - 1, :], X_cascade=X_cascade[t + horizons - 1, :],
                            nsamps=nsamps, t_dist=True, nu=nu)

        if ret.__contains__('new_latent_factors'):
            if t >= forecast_start and t <= forecast_end:
                for lf in new_latent_factors:
                    lf.generate_lf_forecast(date=dates.iloc[t], mod=mod, X_transaction=X_transaction[t + horizons - 1, :],
                                                X_cascade = X_cascade[t + horizons - 1, :],
                                                k=k, nsamps=nsamps, horizons=horizons)
        # Update the DBCM
        if is_lf:
            if isinstance(latent_factor, (list, tuple)):
                pm_bern, ps_bern = latent_factor[0].get_lf(dates.iloc[t])
                pm_pois, ps_pois = latent_factor[1].get_lf(dates.iloc[t])
                pm = (pm_bern, pm_pois)
                ps = (ps_bern, ps_pois)
            else:
                pm, ps = latent_factor.get_lf(dates.iloc[t])

            mod.update_lf_analytic(y_transaction=Y_transaction[t], X_transaction=X_transaction[t, :],
                                   y_cascade=Y_cascade[t,:], X_cascade=X_cascade[t, :],
                                   phi_mu=pm, phi_sigma=ps, excess=excess[t])
        else:
            mod.update(y_transaction=Y_transaction[t], X_transaction=X_transaction[t, :],
                       y_cascade=Y_cascade[t,:], X_cascade=X_cascade[t, :], excess=excess[t])

        if ret.__contains__('new_latent_factors'):
            for lf in new_latent_factors:
                lf.generate_lf(date=dates.iloc[t], mod=mod, X_transaction=X_transaction[t + horizons - 1, :],
                                   X_cascade = X_cascade[t + horizons - 1, :],
                                   k=k, nsamps=nsamps, horizons=horizons)

    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
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