import sys
sys.path.insert(0,'../')
import numpy as np
import pandas as pd
from forecasting.define_models import define_normal_dlm
from forecasting.analysis import analysis_lognormal_seasonalms, analysis_dbcm
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
Y_total = np.log(1 + data['Y_totalsales'])
X_total = data['X_totalsales']
excess = list(data['excess'])
sales = data['sales']
T = len(Y_transaction)
prior_length = 21

# Set parameters
period = 7 # Period of the seasonal component that is coming from the log-normal
k = 14 # Number of days ahead that we will forecast
rho = 1 # Random effect discount factor to increase variance of forecast distribution

# Define period to forecast over
forecast_start = prior_length + 150
forecast_end = T - k
nsamps = 500

# Get multiscale signal from higher level log-normal model
period = 7 # Include a day-of-week seasonal factor - This coefficient will be the multiscale factor we care about
phi_mu_prior, phi_sigma_prior, phi_mu_post, phi_sigma_post = analysis_lognormal_seasonalms(
    Y_total, X_total, prior_length, k, forecast_start, forecast_end, period)

# Update and forecast the model
forecast_samples = analysis_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade, excess, prior_length,
                               k, forecast_start, forecast_end, nsamps, rho,
                               phi_mu_prior, phi_sigma_prior, None, phi_mu_post, phi_sigma_post, delregn=.98)

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


filename = "dbcm_multiscale_forecast"
time = np.arange(forecast_start, forecast_end)
plot_sales_forecast(forecast_samples[:,:,0], sales[forecast_start:forecast_end], time, filename)
