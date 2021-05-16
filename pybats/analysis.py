# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_analysis.ipynb (unless otherwise specified).

__all__ = ['analysis', 'analysis_dcmm', 'analysis_dbcm', 'analysis_dlmm']

# Internal Cell
#exporti
import numpy as np
import pandas as pd

from .define_models import define_dglm, define_dcmm, define_dbcm, define_dlmm
from .shared import define_holiday_regressors
from collections.abc import Iterable

# Cell
def analysis(Y, X=None, k=1, forecast_start=0, forecast_end=0,
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
    T = len(Y) + 1 #np.min([len(Y), forecast_end]) + 1

    if ret.__contains__('model_coef'):
        m = np.zeros([T-1, mod.a.shape[0]])
        C = np.zeros([T-1, mod.a.shape[0], mod.a.shape[0]])
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
        X = np.array([None]*(T+k)).reshape(-1,1)
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
            #for lf in new_latent_factors:
            #    lf.append_lf()
            #    lf.append_lf_forecast()
            if len(new_latent_factors) == 1:
                out.append(new_latent_factors[0])
            else:
                out.append(new_latent_factors)

    if len(out) == 1:
        return out[0]
    else:
        return out

# Cell
def analysis_dcmm(Y, X, k=1, forecast_start=0, forecast_end=0,
                  nsamps=500, rho=.6,
                  model_prior=None, prior_length=20, ntrend=1,
                  dates=None, holidays=[],
                  seasPeriods=[], seasHarmComponents=[],
                  latent_factor=None, new_latent_factors=None,
                  mean_only=False,
                  ret=['model', 'forecast'],
                   **kwargs):
    """
    This is a helpful function to run a standard analysis using a DCMM.
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
        X = define_holiday_regressors(X, dates, holidays)

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

    T = len(Y) + 1 # np.min([len(Y), forecast_end]) + 1
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
        if t < len(Y):
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
            #for lf in new_latent_factors:
            #    lf.append_lf()
            #    lf.append_lf_forecast()
            if len(new_latent_factors) == 1:
                out.append(new_latent_factors[0])
            else:
                out.append(new_latent_factors)

    if len(out) == 1:
        return out[0]
    else:
        return out

# Cell
def analysis_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade, excess,
                  k, forecast_start, forecast_end, nsamps = 500, rho = .6,
                  model_prior=None, prior_length=20, ntrend=1,
                  dates=None, holidays = [],
                  latent_factor = None, new_latent_factors = None,
                  seasPeriods = [], seasHarmComponents = [],
                  mean_only=False,
                  ret=['model', 'forecast'],
                   **kwargs):
    """
    This is a helpful function to run a standard analysis using a DBCM.
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

    T = len(Y_transaction) + 1 #np.min([len(Y_transaction)- k, forecast_end]) + 1
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
        if t < len(Y_transaction):
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
            #for lf in new_latent_factors:
            #    lf.append_lf()
            #    lf.append_lf_forecast()
            if len(new_latent_factors) == 1:
                out.append(new_latent_factors[0])
            else:
                out.append(new_latent_factors)

    if len(out) == 1:
        return out[0]
    else:
        return out

# Cell
def analysis_dlmm(Y, X, prior_length, k, forecast_start, forecast_end,
                  nsamps=500, rho=.6,
                  mean_only=False, dates=None, holidays=[],
                  seasPeriods=[], seasHarmComponents=[], ret=['model', 'forecast'],
                  **kwargs):
    """
    This is a helpful function to run a standard analysis using a DLMM.
    """

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


    # Initialize the DLMM
    if not kwargs.__contains__('model_prior'):
#         a0_dlm = np.zeros(X.shape[1] + 1)
#         R0_dlm = np.eye(X.shape[1] + 1)
#         a0_bern = np.zeros(X.shape[1] + 1)
#         R0_bern = np.eye(X.shape[1] + 1)
#         mod = define_dlmm(Y, X, a0_dlm = a0_dlm, R0_dlm = R0_dlm, a0_bern = a0_bern, R0_bern = R0_bern,
#                           seasPeriods = seasPeriods, seasHarmComponents = seasHarmComponents,
#                           nlf = nlf, rho = rho, nhol = nhol, **kwargs)
        mod = define_dlmm(Y, X, prior_length = prior_length, seasPeriods = seasPeriods, seasHarmComponents = seasHarmComponents,
                          nlf = nlf, rho = rho, nhol = nhol, **kwargs)
    else:
        mod = kwargs.get('model_prior')

    # Initialize updating + forecasting
    horizons = np.arange(1,k+1)

    if mean_only:
        forecast = np.zeros([1, forecast_end - forecast_start + 1, k])
    else:
        forecast = np.zeros([nsamps, forecast_end - forecast_start + 1, k])

    T = np.min([len(Y), forecast_end]) + 1
    nu = 9

    if ret.__contains__('model_coef'): ## Return normal dlm params
        m = np.zeros([T, mod.dlm_mod.a.shape[0]])
        C = np.zeros([T, mod.dlm_mod.a.shape[0], mod.dlm_mod.a.shape[0]])
        a = np.zeros([T, mod.dlm_mod.a.shape[0]])
        R = np.zeros([T, mod.dlm_mod.a.shape[0], mod.dlm_mod.a.shape[0]])
        n = np.zeros(T)
        s = np.zeros(T)



    # Run updating + forecasting
    for t in range(prior_length, T):
        if ret.__contains__('forecast'):
            if t >= forecast_start and t <= forecast_end:
                if t == forecast_start:
                    print('beginning forecasting')

                # Get the forecast samples for all the items over the 1:k step ahead path
                if mean_only:
                    forecast[:, t - forecast_start, :] = np.array(list(map(
                        lambda k, x: mod.forecast_marginal(
                            k=k, X=(x, x), nsamps=nsamps, mean_only=mean_only),
                        horizons, X[t + horizons - 1, :]))).reshape(1,-1)
                else:
                    forecast[:, t - forecast_start, :] = mod.forecast_path(
                    k=k, X=(X[t + horizons - 1, :], X[t + horizons - 1, :]), nsamps=nsamps)


        mod.update(y = Y[t], X=(X[t], X[t]))
        if ret.__contains__('model_coef'):
            m[t,:] = mod.dlm_mod.m.reshape(-1)
            C[t,:,:] = mod.dlm_mod.C
            a[t,:] = mod.dlm_mod.a.reshape(-1)
            R[t,:,:] = mod.dlm_mod.R
            n[t] = mod.dlm_mod.n / mod.dlm_mod.delVar
            s[t] = mod.dlm_mod.s

    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
        if obj == 'model_coef':
            mod_coef = {'m':m, 'C':C, 'a':a, 'R':R, 'n':n, 's':s}
            out.append(mod_coef)

    if len(out) == 1:
        return out[0]
    else:
        return out