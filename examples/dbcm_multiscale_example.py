import sys
sys.path.insert(0,'../')

import numpy as np
import pandas as pd
from PyBATS.define_models import define_normal_dlm
from PyBATS.analysis import analysis_dlm, analysis_dbcm
from PyBATS.latent_factor import seas_weekly_lf
import matplotlib.pyplot as plt

## Load in data:
### Y_totalsales = total sales of a type of item (proxy for overall store traffic)
### sales = sales of a single item
### transaction - total number of transactions
### cascade - counts of basket size per transaction
### excess - baskets with more items than the number of cascades (4)
data = np.load("../PyBATS/data/dbcm_multiscale_data.npz")
Y_transaction = data['Y_transaction']
X_transaction = data['X_transaction']
Y_cascade = data['Y_cascade']
X_cascade = data['X_cascade']
Y_total = np.log(1 + data['Y_totalsales'])
X_total = data['X_totalsales']
excess = list(data['excess'])
sales = data['sales']
T = len(Y_transaction)
# Make up a start date
start_date = pd.to_datetime('2017-01-01')
dates = pd.date_range(start_date, start_date + pd.DateOffset(days=T) , freq='D')
prior_length = 21

# Set parameters
k = 14 # Number of days ahead that we will forecast
rho = 1 # Random effect discount factor to increase variance of forecast distribution
nsamps = 500

# Define period to forecast over
forecast_start = prior_length + 150
forecast_start_date = start_date + pd.DateOffset(days=forecast_start)
forecast_end_date = dates[-1] - pd.DateOffset(days=k)


# Get the multiscale signal (a latent factor) from higher level log-normal model
latent_factor = analysis_dlm(Y_total, X_total, prior_length, k, forecast_start_date, forecast_end_date, dates=dates,
                               ret=['new_latent_factors'], new_latent_factors= [seas_weekly_lf.copy()])

# Update and forecast the model
forecast_samples = analysis_dbcm(Y_transaction, X_transaction, Y_cascade, X_cascade, excess, prior_length,
                                 k, forecast_start_date, forecast_end_date, nsamps, rho,
                                 latent_factor, dates = dates, delregn=.98)

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


filename = "./Examples/dbcm_multiscale_forecast"
forecast_end = T - k + 1
time = np.arange(forecast_start, forecast_end)
plot_sales_forecast(forecast_samples[:,:,0], sales[forecast_start:forecast_end], dates[forecast_start:forecast_end], filename)
