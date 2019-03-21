import sys
sys.path.insert(0,'../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecasting.define_models import define_dcmm, define_normal_dlm
from forecasting.multiscale import get_latent_factor, forecast_latent_factor, sample_latent_factor


## Load in data:
### Y_totalsales = total sales of a type of item (proxy for overall store traffic)
### Y = sales of a single item
data = np.load("../data/dcmm_multiscale_data.npz")
Y = data['Y'].T
X = data['X']
Y_total = np.log(data['Y_totalsales'])
X_total = data['X_totalsales']
T = len(Y)
prior_length = 21

## Define normal DLM for the log of total sales
## Include a day-of-week seasonal factor - This coefficient will be the multiscale factor we care about
totalsales_mod = define_normal_dlm(Y_total, prior_length)
period = totalsales_mod.seasPeriods[0]


## Define multiscale DCMM
rho = .5
dcmm_multiscale = define_dcmm(Y, X, prior_length = prior_length, multiscale = True, rho = rho, delpois=.99, delbern=.99)

#Initialize parameters and storage variables
k = 14 # Number of days ahead that we will forecast
horizons = np.arange(1,k+1)
forecast_start = prior_length + 200
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

        # Get the forecast 1:14-steps ahead with the multiscale DCMM
        forecast_samps[:, t - forecast_start, :] = np.array(list(map(lambda k, x, lf:
                                                                     dcmm_multiscale.multiscale_forecast_marginal_approx(
                                                                       k,
                                                                       X = (x, x),
                                                                       phi_mu=(lf[0], lf[0]),
                                                                       phi_sigma = (lf[1], lf[1]),
                                                                       nsamps = nsamps),
                                                                     horizons,
                                                                     X[t + horizons - 1],
                                                                     future_latent_factors))).T

    # Now observe the true y value, and update:

    # Update the normal DLM for total sales
    totalsales_mod.update(Y_total[t], X_total[t])

    # Get posterior mean and variance of the latent factors
    phi_mu, phi_sigma = get_latent_factor(totalsales_mod, day=today)
    # Update a multiscale DCMM of a single item's sales
    dcmm_multiscale.multiscale_update_approx(y=Y[t], X=(X[t], X[t]),
                                             phi_mu=(phi_mu, phi_mu), phi_sigma=(phi_sigma, phi_sigma))

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
    plt.savefig(filename+'.jpg', dpi=300)


filename = "dcmm_multiscale_forecast2"
time = np.arange(forecast_start, forecast_end)
plot_sales_forecast(forecast_samps[:,:,0], Y[forecast_start:forecast_end], time, filename)