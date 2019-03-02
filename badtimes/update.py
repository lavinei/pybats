import numpy as np


def update(mod, y = None, X = None):
    if mod.nregn > 0:
        mod.F[mod.iregn] = X
                    
    # If data is missing then skip discounting and updating, posterior = prior
    if y is None or np.isnan(y):
        mod.t += 1
        mod.m = mod.a
        mod.C = mod.R
        
        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T)/2
        
        mod.W = mod.get_W()
            
    else:
            
        # Mean and variance
        ft, qt = mod.get_mean_and_var(mod.F, mod.a, mod.R)

        # Choose conjugate prior, match mean and variance
        mod.param1, mod.param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)
            
        # See time t observation y (which was passed into the update function)
        mod.t += 1
        
        # Update the conjugate parameters and get the implied ft* and qt*
        mod.param1, mod.param2, ft_star, qt_star = mod.update_conjugate_params(y, mod.param1, mod.param2)
            
        # Kalman filter update on the state vector (using Linear Bayes approximation)
        mod.m = mod.a + mod.R @ mod.F * (ft_star - ft)/qt
        mod.C = mod.R - mod.R @ mod.F @ mod.F.T @ mod.R * (1 - qt_star/qt)/qt
        
        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T)/2
                
        # Discount information in the time t + 1 prior
        mod.W = mod.get_W()
        mod.R = mod.R + mod.W


def update_bindglm(mod, n=None, y=None, X=None):
    if mod.nregn > 0:
        mod.F[mod.iregn] = X

    # If data is missing then skip discounting and updating, posterior = prior
    if y is None or np.isnan(y) or n is None or n == 0:
        mod.t += 1
        mod.m = mod.a
        mod.C = mod.R

        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T) / 2

        mod.W = mod.get_W()

    else:

        # Mean and variance
        ft, qt = mod.get_mean_and_var(mod.F, mod.a, mod.R)

        # Choose conjugate prior, match mean and variance
        mod.param1, mod.param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)

        # See time t observation y (which was passed into the update function)
        mod.t += 1

        # Update the conjugate parameters and get the implied ft* and qt*
        mod.param1, mod.param2, ft_star, qt_star = mod.update_conjugate_params(n, y, mod.param1, mod.param2)

        # Kalman filter update on the state vector (using Linear Bayes approximation)
        mod.m = mod.a + mod.R @ mod.F * (ft_star - ft) / qt
        mod.C = mod.R - mod.R @ mod.F @ mod.F.T @ mod.R * (1 - qt_star / qt) / qt

        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T) / 2

        # Discount information in the time t + 1 prior
        mod.W = mod.get_W()
        mod.R = mod.R + mod.W


def update_normaldlm(mod, y = None, X = None):
    if mod.nregn > 0:
        mod.F[mod.iregn] = X
            
    # If data is missing then skip discounting and updating, posterior = prior
    if y is None or np.isnan(y):
        mod.t += 1
        mod.m = mod.a
        mod.C = mod.R
        
        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T)/2
        
        mod.W = mod.get_W()
            
    else:
            
        # Mean and variance
        ft, qt = mod.get_mean_and_var(mod.F, mod.a, mod.R)
        mod.param1 = ft
        mod.param2 = qt
        
        # See time t observation y (which was passed into the update function)
        mod.t += 1
        
        # Update the  parameters:
        et = y - ft

        # Adaptive coefficient vector
        At = mod.R @ mod.F / qt
        
        # Volatility estimate ratio
        rt = (mod.n + et**2/qt)/(mod.n + 1)
        
        # Kalman filter update
        mod.n = mod.n + 1
        mod.s = mod.s * rt
        mod.m = mod.a + At * et
        mod.C = rt * (mod.R - qt * At @ At.T)
        
        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T)/2
        
        # Discount information
        mod.W = mod.get_W()
        mod.R = mod.R + mod.W
        mod.n = mod.delVar * mod.n