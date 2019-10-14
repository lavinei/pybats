import numpy as np
import matplotlib.pyplot as plt
from pybats.analysis import analysis
from pybats.point_forecast import median
from pybats.plot import plot_data_forecast, ax_style
from pybats.shared import load_sales_example

data = load_sales_example()                                 # Load example sales and advertising data. Source: Abraham & Ledolter (1983)
Y = data['Sales'].values
X = data['Advertising'].values

k = 1                                                       # Forecast 1 step ahead
forecast_start = 15                                         # Start forecast at time step 15
forecast_end = 35                                           # End forecast at time step 35 (final time step)

# Run a simple analysis
mod, samples = analysis(Y, X, family="poisson",
                        forecast_start=forecast_start,    # First time step to forecast on
                        forecast_end=forecast_end,          # Final time step to forecast on
                        k=k,                              # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
                        prior_length=6,                     # How many data point to use in defining prior
                        rho=.5,                           # Random effect extension, increases variance of Poisson DGLM (see Berry and West, 2019)
                        deltrend=0.95,                      # Discount factor on the trend component (intercept)
                        delregn=0.95                        # Discount factor on the regression component
                        )

forecast = median(samples)                                  # Take the median as the point forecast


fig, ax = plt.subplots(1,1)                                 # Plot the 1-step ahead point forecast plus the 95% credible interval
ax = plot_data_forecast(fig, ax, Y[forecast_start:forecast_end + k], forecast, samples,
                        dates=np.arange(forecast_start, forecast_end+1, dtype='int'))
ax = ax_style(ax, ylabel='Sales', xlabel='Time', xlim=[forecast_start, forecast_end],
              legend=['Forecast', 'Sales', 'Credible Interval'])
plt.savefig('./forecast.jpg')