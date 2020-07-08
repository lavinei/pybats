import numpy as np
import pandas as pd
import scipy as sc
import pickle
from scipy.special import digamma
from pandas.tseries.holiday import AbstractHolidayCalendar, USMartinLutherKingJr, USMemorialDay, Holiday, USLaborDay, \
    USThanksgivingDay
import os

# I need this helper in a module file for pickle reasons ...
def transformer(ft, qt, fn1, fn2):
    return np.exp(np.ravel(fn1(ft, np.sqrt(qt), grid=False))), \
           np.exp(np.ravel(fn2(ft, np.sqrt(qt), grid=False)))


def gamma_transformer(ft, qt, fn):
    alpha = np.ravel(np.exp(fn(np.sqrt(qt))))
    beta = np.exp(digamma(alpha) - ft)
    return alpha, beta


def trigamma(x):
    return sc.special.polygamma(x=x, n=1)


def save(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file=file)


def load(filename):
    with open(filename, "rb") as file:
        tmp = pickle.load(file)
    return tmp


def define_holiday_regressors(X, dates, holidays=None):
    """
    Add columns to the predictor matrix X for a specified list of holidays

    :param X: (Array) Predictor matrix without columns for the holidays
    :param dates: Dates
    :param holidays: (List) holidays
    :return: Updated predictor matrix
    """
    if holidays is not None:
        if len(holidays) > 0:
            if X is None:
                n = len(dates)
            else:
                n = X.shape[0]

            for holiday in holidays:
                cal = AbstractHolidayCalendar()
                cal.rules = [holiday]
                x = np.zeros(n)
                x[dates.isin(cal.holidays())] = 1
                if X is None:
                    X = x
                else:
                    X = np.c_[X, x]

            return X
        else:
            return X
    else:
        return X


def cov2corr(cov):
    """
    Transform a covariance matrix into correlation. Useful to understand the state vector correlation (mod.C)

    :param cov: Covariance matrix
    :return: Correlation matrix
    """
    D = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return cov / D / D.T

def load_sales_example():
    """
    Read data for the first sales forecasting example

    :return: A Pandas data frame
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data/'
    return pd.read_csv(data_dir + 'sales.csv', index_col=0)

def load_sales_example2():
    """
    Read data for the second sales forecasting example

    :return: A Pandas data frame
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data/'
    return pd.read_pickle(data_dir + 'sim_sales_data')

def load_dcmm_latent_factor_example():
    """
    Read data for the DCMM latent factor example

    :return: A list of data
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data/'
    data = load(data_dir + 'dcmm_latent_factor_data')
    return data

def load_dbcm_latent_factor_example():
    """
    Read data for the DCMM latent factor example

    :return: A list of data
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data/'
    data = load(data_dir + 'dbcm_latent_factor_data')
    return data


def load_standard_holidays():
    holidays = [USMartinLutherKingJr,
                USMemorialDay,
                Holiday('July4', month=7, day=4),
                USLaborDay,
                # Holiday('Thanksgiving_1DB', month=11, day=1, offset=pd.DateOffset(weekday=WE(4))),
                USThanksgivingDay,
                # Holiday('Christmas_1DB', month=12, day=24),
                Holiday('Christmas', month=12, day=25),
                Holiday('New_Years_Eve', month=12, day=31),
                ]
    return holidays
