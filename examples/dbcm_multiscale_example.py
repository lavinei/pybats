import sys
sys.path.insert(0,'../')
import numpy as np
import pandas as pd
from badtimes.define_models import define_dbcm, define_normal_dlm
from badtimes.multiscale import get_latent_factor, forecast_latent_factor, sample_latent_factor
from badtimes.update import update_normaldlm
import matplotlib.pyplot as plt

## Load in data:
### Y_totalsales = total sales of a type of item (proxy for overall store traffic)
### sales = sales of a single item
### transaction - total number of transactions
### cascade - counts of basket size per transaction
### excess - baskets with more items than the number of cascades (4)
data = np.load("../data/dbcm_multiscale_data.npz")
Y_transaction = data['Y_transaction']
X_transaction = data['X_transaction']
Y_cascade = data['Y_cascade']
X_cascade = data['X_cascade']
Y_total = np.log(data['Y_totalsales'])
X_total = data['X_totalsales']
excess = list(data['excess'])
sales = data['sales']
T = len(Y_transaction)
prior_length = 21


## Define normal DLM for the log of total sales
## Include a day-of-week seasonal factor - This coefficient will be the multiscale factor we care about
totalsales_mod = define_normal_dlm(Y_total, prior_length)
period = totalsales_mod.seasPeriods[0]

## Define multiscale DBCM
rho = 0.4
dbcm_multiscale = define_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade, excess_values = excess, multiscale=True,
                              rho = rho, deltrend=.98, delregn = .99, delmultiscale=.99, delseas=.99, prior_length= prior_length)


k = 14 # Number of days ahead that we will forecast
horizons = np.arange(1,k+1)
forecast_start = prior_length + 150
forecast_end = T - k
nsamps = 500
forecast_samps = np.zeros([nsamps, forecast_end - forecast_start, k])

# Now update and forecast
for t in range(prior_length, T):
    # Get the day-of-week
    today = t % period

    if t % 100 == 0:
        print(t)

    if t >= forecast_start and t < forecast_end:

        # Forecast the mean and variance of the latent factor 1:k steps ahead
        future_latent_factors = list(map(lambda k: forecast_latent_factor(totalsales_mod, k=k, today=today, period=period),
                                         horizons))
        phi_mu = [lf[0] for lf in future_latent_factors]
        phi_sigma = [lf[1] for lf in future_latent_factors]

        # Get the forecast 1:14-steps ahead with the multiscale DCMM
        forecast_samps[:, t - forecast_start, :] = dbcm_multiscale.multiscale_forecast_path_approx(
            k,
            X_transaction[t + horizons - 1],
            X_cascade[t + horizons - 1],
            phi_mu = phi_mu,
            phi_sigma = phi_sigma,
            nsamps = nsamps
                                                                                                   )

    # Now observe the true y value, and update:

    # Update the normal DLM for total sales
    totalsales_mod.update(Y_total[t], X_total[t])

    # Get posterior mean and variance of the latent factors
    phi_mu, phi_sigma = get_latent_factor(totalsales_mod, day=today)
    # Update a multiscale DCMM of a single item's sales
    dbcm_multiscale.multiscale_update_approx(Y_transaction[t], X_transaction[t],
                                             Y_cascade[t,:], X_cascade[t],
                                             phi_mu=phi_mu, phi_sigma=phi_sigma,
                                             excess = excess[t])

dbcm_multiscale.dcmm.pois_mod.a
dbcm_multiscale.dcmm.pois_mod.param1
np.round(dbcm_multiscale.dcmm.pois_mod.R, 2)

## Plot forecasts against true sales, along with 95% credible intervals
def plot_sales_forecast(forecast_samps, sales, time, filename):
    plot_data = np.c_[np.mean(forecast_samps, axis=0).reshape(-1), sales]
    plt.figure(figsize=(10, 6))
    plt.plot(time, plot_data[:, 0], color='b')
    plt.scatter(time, plot_data[:, 1], color='k')
    upper = np.percentile(forecast_samps, [97.5], axis=0).reshape(-1)
    lower = np.percentile(forecast_samps, [2.5], axis=0).reshape(-1)
    plt.fill_between(time, upper, lower, alpha=.3)
    plt.legend(["Forecast Mean", "Sales", "95% Credible Interval"])
    plt.ylabel('Sales')
    plt.xlabel('Time')
    plt.ylim(0,150)
    plt.savefig(filename+'.jpg', dpi=300)


filename = "dbcm_multiscale_forecast"
time = np.arange(forecast_start, forecast_end)
plot_sales_forecast(forecast_samps[:,:,0], sales[forecast_start:forecast_end], time, filename)