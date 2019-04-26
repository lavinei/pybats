from setuptools import setup

setup(
    name="forecasting",
    version="0.0.1",
    description="Bayesian Forecasting of Discrete Time Series",
    author="Isaac Lavine",
    packages=['forecasting'],
    package_dir={'forecasting': 'forecasting'},
    package_data={'forecasting': ['pkg_data/*.pickle.gzip']}
)
