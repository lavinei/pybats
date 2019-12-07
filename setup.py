from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pybats",
    version="0.0.1",
    description="Bayesian Forecasting of Time Series",
    author="Isaac Lavine",
    author_email="lavine.isaac@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://lavinei.github.io/pybats/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ], install_requires=['pandas', 'numpy', 'statsmodels', 'scipy'],
    package_dir={'pybats': 'pybats'},
    package_data={'pybats': ['pkg_data/*']},
    python_requires='>=3.6'#'pkg_data/*.pickle.gzip', 'pkg_data/sales.csv',
)
