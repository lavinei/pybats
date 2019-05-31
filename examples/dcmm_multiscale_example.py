import sys
sys.path.insert(0,'../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyBATS.analysis import analysis_dlm, analysis_dcmm

## Load in data:
### Y_totalsales = total sales of a type of item (proxy for overall store traffic)
### Y = sales of a single item
data = np.load("../PyBATS/data/dcmm_multiscale_data.npz")
Y = data['Y'].T
X = data['X'].reshape(-1,1)
Y_total = np.log(data['Y_totalsales'])
X_total = data['X_totalsales']
T = len(Y)
prior_length = 21


#Initialize parameters
period = 7 # Period of the seasonal component that is coming from the log-normal
rho = .5
k = 14 # Number of days ahead that we will forecast
horizons = np.arange(1,k+1)
forecast_start = prior_length + 200
forecast_end = T - k
nsamps = 500

# Get multiscale signal from higher level log-normal model
phi_mu_prior, phi_sigma_prior, phi_mu_post, phi_sigma_post = analysis_dlm(
    Y_total, X_total, prior_length, k, forecast_start, forecast_end, period)


# Update and forecast the model
forecast_samples = analysis_dcmm(Y, X, prior_length,
                               k, forecast_start, forecast_end, nsamps, rho,
                               phi_mu_prior, phi_sigma_prior, None, phi_mu_post, phi_sigma_post)


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


filename = "./Examples/dcmm_multiscale_forecast"
time = np.arange(forecast_start, forecast_end)
plot_sales_forecast(forecast_samples[:,:,0], Y[forecast_start:forecast_end], time, filename)