from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PyBATS",
    version="0.0.1",
    description="Bayesian Forecasting of Time Series",
    author="Isaac Lavine",
    author_email="lavine.isaac@gmail.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
    package_dir={'PyBATS': 'PyBATS'},
    package_data={'PyBATS': ['pkg_data/*.pickle.gzip']}
)
