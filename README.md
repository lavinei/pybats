# PyBATS

PyBATS is a package for Bayesian time series modeling and forecasting. It is designed to be flexible, offering many options to customize the model form, prior, and forecast period. The focus of the package is the class Dynamic Generalized Linear Model ('dglm'). The supported DGLMs are Poisson, Bernoulli, Normal (a DLM), and Binomial. These models are based upon *Bayesian Forecasting and Dynamic Models*, by West and Harrison (1997).

## Installation
PyBATS is in development, and is currently hosted on [GitHub](https://github.com/lavinei/pybats). You can download and install from there:

```
$ git clone git@github.com:lavinei/pybats.git pybats
$ cd pybats
$ sudo python setup.py install
```

## Quick Start
This provides the most basic example of Bayesian time series analysis using PyBATS. We'll use a public dataset of the sales of a dietary weight control product, along with the advertising spend. These are integer valued counts, which we model with a Poisson Dynamic Generalized Linear Model (DGLM).

First we load in the data, and take a quick look at the first couples of entries:
```
import numpy as np
from pybats.shared import load_sales_example

data = load_sales_example()                                 # Load example sales and advertising data. Source: Abraham & Ledolter (1983)
print(data[:5])
```

Second, we extract the outcome ($Y$) and covariate ($X$) from this dataset. We'll set the forecast horizon $k=1$ for this example. We could look at multiple forecast horizons by setting $k$ to a larger value. Then the 'analysis' function will automatically perform marginal forecasts across horizons $1:k$.

Finally, we set the start and end time for forecasting. In this case we specify the start and end date with intergers, because there are no dates associated with this dataset.
```
Y = data['Sales'].values
X = data['Advertising'].values

k = 1                                                       # Forecast 1 step ahead
forecast_start = 15                                         # Start forecast at time step 15
forecast_end = 35                                           # End forecast at time step 35 (final time step)
```

We use the *analysis* function as a helper to a) define the model b) Run sequential updating (forward filtering) and c) forecasting. By default, it will return samples from the forecast distribution as well as the model after the final observation.
```
from pybats.analysis import analysis

mod, samples = analysis(Y, X, family="poisson",
forecast_start=forecast_start,    # First time step to forecast on
forecast_end=forecast_end,          # Final time step to forecast on
k=k,                              # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
prior_length=6,                     # How many data point to use in defining prior
rho=.5,                           # Random effect extension, increases variance of Poisson DGLM (see Berry and West, 2019)
deltrend=0.95,                      # Discount factor on the trend component (intercept)
delregn=0.95                        # Discount factor on the regression component
)
```

The model has the posterior mean and variance of the state vector stored as ```mod.a``` and ```mod.C``` respectively.  In this example, we are purely interested in the forecasts. We'll take the median as the point forecast, and also plot the $95\%$ credible interval.
```
import matplotlib.pyplot as plt
from pybats.point_forecast import median
from pybats.plot import plot_data_forecast, ax_style

forecast = median(samples)                                  # Take the median as the point forecast


fig, ax = plt.subplots(1,1)                                 # Plot the 1-step ahead point forecast plus the 95% credible interval
ax = plot_data_forecast(fig, ax, Y[forecast_start:forecast_end + k], forecast, samples,
                        dates=np.arange(forecast_start, forecast_end+1, dtype='int'))
ax = ax_style(ax, ylabel='Sales', xlabel='Time', xlim=[forecast_start, forecast_end])
plt.savefig('./forecast.jpg')
```

The resulting forecast image is:
![forecast](examples/forecast.jpg "1-step Forecasts and Credible Intervals")

References:
1. Berry, L., West, M., 2018. Bayesian forecasting of many count-valued time series. Submitted for publication. ArXiv:1805.05232

2. Berry, L., Helman, P., West, M., 2018. Probabilistic forecasting of heterogeneous consumer transaction-sales time series. Submitted for publication. arXiv:1808.04698

3. West, M., Harrison, J., 1997. Bayesian Forecasting and Dynamic Models, 2nd Edition. Springer-Verlag,
New York, Inc.
