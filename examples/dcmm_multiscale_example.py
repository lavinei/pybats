import sys
sys.path.insert(0,'../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyBATS.analysis import analysis_dlm, analysis_dcmm
from PyBATS.latent_factor import seas_weekly_lf

## Load in data:
### Y_totalsales = total sales of a type of item (proxy for overall store traffic)
### Y = sales of a single item
data = np.load("../PyBATS/data/dcmm_multiscale_data.npz")
Y = data['Y'].T
X = data['X'].reshape(-1,1)
Y_total = np.log(data['Y_totalsales']).reshape(-1,1)
X_total = data['X_totalsales'].reshape(-1,1)
T = len(Y)
start_date = pd.to_datetime('2017-01-01') # Make up a start date
dates = pd.date_range(start_date, start_date + pd.DateOffset(days=T) , freq='D')
prior_length = 21

#Initialize parameters
rho = .5
k = 14 # Number of days ahead that we will forecast
horizons = np.arange(1,k+1)
nsamps = 500

# Define period to forecast over
forecast_start = prior_length + 150
forecast_start_date = start_date + pd.DateOffset(days=forecast_start)
forecast_end_date = dates[-1] - pd.DateOffset(days=k)

# Get multiscale signal (a latent factor) from higher level log-normal model
latent_factor = analysis_dlm(Y_total, X_total, prior_length, k, forecast_start_date, forecast_end_date, dates=dates,
                             ret=['new_latent_factors'], new_latent_factors= [seas_weekly_lf])


# Update and forecast the model
forecast_samples = analysis_dcmm(Y, X, prior_length, k, forecast_start_date, forecast_end_date, nsamps, rho,
                                 latent_factor, dates=dates)


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
forecast_end = T - k + 1
time = np.arange(forecast_start, forecast_end)
plot_sales_forecast(forecast_samples[:,:,0], Y[forecast_start:forecast_end], dates[forecast_start:forecast_end], filename)
