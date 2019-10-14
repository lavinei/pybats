import sys
sys.path.insert(0,'../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybats.define_models import define_dglm
from pybats.analysis import analysis
from pybats.point_forecast import median
from pybats.plot import plot_data_forecast


data = pd.read_csv('../data/sales.csv', index_col=0)
Y = data['Sales'].values
X = data['Advertising'].values

k = 1
forecast_start = 15
forecast_end = 35
forecast_len = forecast_end + k - forecast_start


mod, samples = analysis(Y, X, family="poisson",
                        forecast_start = forecast_start, forecast_end=forecast_end, k = k,
                        prior_length=6,
                        rho = .5)

forecast = median(samples)

fig, ax = plt.subplots(1,1)
ax = plot_data_forecast(fig, ax, Y[forecast_start:forecast_end + k], forecast, samples, dates=np.arange(forecast_len))
plt.savefig('./forecast.jpg')