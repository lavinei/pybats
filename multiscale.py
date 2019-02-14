# These are for the general DGLM
import numpy as np
import scipy as sc
from forecasting.seasonal import *
from forecasting.forecast import *

def multiscale_update(mod, y = None, X = None, phi_samps = None):
    """
    phi_samps: array of samples of the latent factor vector
    """
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
        # Update m, C using a weighted average of the samples
        output = map(lambda p: multiscale_update_with_samp(mod, y, mod.F, mod.a, mod.R, p), phi_samps)
        mlist, Clist, logliklist = list(map(list, zip(*output)))
        w = (np.exp(logliklist) / np.sum(np.exp(logliklist))).reshape(-1,1,1)
        mlist = np.array(mlist)
        Clist = np.array(Clist)
        mod.m = np.sum(mlist*w, axis=0)
        mod.C = np.sum(Clist*w, axis=0) + np.cov((mlist).reshape(-1, mod.m.shape[0]), rowvar=False, aweights = w.reshape(-1))
            
        # Add 1 to the time index
        mod.t += 1
        
        # Get priors a, R from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T)/2 # prevent rounding issues
        
        # Discount information if observation is observed
        mod.W = mod.get_W()
        mod.R = mod.R + mod.W
        
                
def multiscale_update_with_samp(mod, y, F, a, R, phi):
    F[mod.imultiscale] = phi.reshape(-1,1)
    ft, qt = mod.get_mean_and_var(F, a, R)
    # get the conjugate prior parameters
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)
    # Get the log-likelihood of 'y' under these parameters
    loglik = mod.loglik(y, param1, param2)
    # Update to the conjugate posterior after observing 'y'
    param1, param2, ft_star, qt_star = mod.update_conjugate_params(y, param1, param2)
    # Kalman filter update on the state vector (using Linear Bayes approximation)
    m = a + R @ F * (ft_star - ft)/qt
    C = R - R @ F @ F.T @ R * (1 - qt_star/qt)/qt
    
    return m, C, loglik
            
def multiscale_update_approx(mod, y = None, X = None, phi_mu = None, phi_sigma = None):
    """
    phi_mu: mean vector of the latent factor
    phi_sigma: variance matrix of the latent factor
        
    Implementing approximation: Assume the latent factor is independent of the state vector
    """
    if mod.nregn > 0:
        mod.F[mod.iregn] = X
            
    # Put the mean of the latent factor phi_mu into the F vector    
    if mod.nmultiscale > 0:
        mod.F[mod.imultiscale] = phi_mu
            
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
        ft, qt = multiscale_get_mean_and_var(mod.F, mod.a, mod.R, phi_mu, phi_sigma, mod.imultiscale)

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
        
        
def multiscale_get_mean_and_var(F, a, R, phi_mu, phi_sigma, imultiscale):
    p = len(imultiscale)
    if p == 1:
        extra_var = a[imultiscale]**2 * phi_sigma + a[imultiscale] * R[np.ix_(imultiscale, imultiscale)] * phi_sigma
    else:
        extra_var = a[imultiscale].T @ phi_sigma @ a[imultiscale] + np.trace(R[np.ix_(imultiscale, imultiscale)] @ phi_sigma)
        
    return F.T @ a, F.T @ R @ F + extra_var
        
    
def multiscale_forecast_marginal(mod, k, X = None, phi_samps = None, mean_only = False):
    """
    Forecast function k steps ahead (marginal)
    """
    # Plug in the correct F values
    if mod.nregn > 0:
        F = np.copy(mod.F)
        F[mod.iregn] = X
        
    Gk = np.linalg.matrix_power(mod.G, k-1)
    a = Gk @ mod.a
    R = Gk @ mod.R @ Gk.T + (k-1)*mod.W

    # Simulate from the forecast distribution
    return np.array(list(map(lambda p: multiscale_sim_with_samp(mod, F, a, R, p), phi_samps))).reshape(-1)
        

def multiscale_sim_with_samp(mod, F, a, R, phi):
    """
    Simulate 'y' values using a single sample of the latent factor phi
    """
    F[mod.imultiscale] = phi.reshape(-1,1)
    ft, qt = mod.get_mean_and_var(F, a, R)
    # get the conjugate prior parameters
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)
    # Update to the conjugate posterior after observing 'y'    
    return mod.simulate(param1, param2, 1)


def multiscale_forecast_marginal_approx(mod, k, X = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False):
    """
    Forecast function k steps ahead (marginal)
    """
    # Plug in the correct F values
    if mod.nregn > 0:
        F = np.copy(mod.F)
        F[mod.iregn] = X
            
    # Put the mean of the latent factor phi_mu into the F vector    
    if mod.nmultiscale > 0:
        F[mod.imultiscale] = phi_mu
        
    Gk = np.linalg.matrix_power(mod.G, k-1)
    a = Gk @ mod.a
    R = Gk @ mod.R @ Gk.T + (k-1)*mod.W
            
    # Mean and variance
    ft, qt = multiscale_get_mean_and_var(F, a, R, phi_mu, phi_sigma, mod.imultiscale)
        
    # Choose conjugate prior, match mean and variance
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)
    
    if mean_only:
        return mod.get_mean(param1, param2)
        
    # Simulate from the forecast distribution
    return mod.simulate(param1, param2, nsamps)
        
def multiscale_forecast_path_approx(mod, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = None, nsamps = 1):  
    """
    Forecast function for the k-step path
    k: steps ahead to forecast
    X: array with k rows for the future regression components
    phi_mu: length k list of mean vectors of the latent factors
    phi_sigma: length k list of variance matrices of the latent factors
    phi_psi: length k list of covariance matrices of phi_t+k and phi_t+j. Each element is another list, of length k.
    nsamps: Number of samples to draw from forecast distribution
    """
    
    lambda_mu = np.zeros([k])
    lambda_cov = np.zeros([k, k])
    
    F = np.copy(mod.F)
                        
    Flist = [None for x in range(k)]
    Rlist = [None for x in range(k)]
    alist = [None for x in range(k)]
    
    for i in range(k):

        # Get the marginal a, R
        Gk = np.linalg.matrix_power(mod.G, i)
        a = Gk @ mod.a
        R = Gk @ mod.R @ Gk.T + (i)*mod.W
            
        alist[i] = a
        Rlist[i] = R

        # Plug in the correct F values
        if mod.nregn > 0:
            F[mod.iregn] = X[i,:]
            
        # Put the mean of the latent factor phi_mu into the F vector    
        if mod.nmultiscale > 0:
            F[mod.imultiscale] = phi_mu[i]
            
        Flist[i] = np.copy(F)
            
        # Find lambda mean and var
        ft, qt = multiscale_get_mean_and_var(F, a, R, phi_mu[i], phi_sigma[i], mod.imultiscale)
        lambda_mu[i] = ft
        lambda_cov[i,i] = qt
        
        # Find covariances with previous lambda values
        for j in range(i):
            # Covariance matrix between the state vector at times j, k
            cov_ij = np.linalg.matrix_power(mod.G, i-j) @ Rlist[j] 
            
            # Covariance between lambda at times j, i
            # If phi_psi is none, we assume the latent factors phi at times t+i, t+j are independent of one another
            if phi_psi is None: 
                lambda_cov[j,i] = lambda_cov[i,j] = Flist[j].T @ cov_ij @ Flist[i]
            else:
                lambda_cov[j,i] = lambda_cov[i,j] = Flist[j].T @ cov_ij @ Flist[i] + alist[i][mod.imultiscale].T @ phi_psi[i][j] @ alist[j][mod.imultiscale]
                                        
    return forecast_path_approx_sim(mod, k, lambda_mu, lambda_cov, nsamps)


################# FUNCTIONS FOR WORKING WITH SEASONAL LATENT FACTORS FROM THE HIGHER LEVEL NORMAL DLM ##############


def get_latent_factor(mod, day):
    phi_mu = np.zeros([mod.seasPeriods[0], 1])
    phi_sigma = np.zeros([mod.seasPeriods[0], mod.seasPeriods[0]])
    phi, var = fourierToSeasonal(mod)
    phi_mu[day] = phi[0]
    phi_sigma[day, day] = var[0, 0]
    return phi_mu, phi_sigma

def get_latent_factor_fxnl(day, L, m, C, iseas, seasPeriods):
    phi_mu = np.zeros([seasPeriods, 1])
    phi_sigma = np.zeros([seasPeriods, seasPeriods])
    phi, var = fourierToSeasonalFxnl(L, m, C, iseas)
    phi_mu[day] = phi[0]
    phi_sigma[day, day] = var[0, 0]
    return phi_mu, phi_sigma

def sample_latent_factor(mod, day, nsamps):
    phi_samps = np.zeros([nsamps, mod.seasPeriods[0]])
    phi, var = fourierToSeasonalFxnl(mod.L, mod.m, mod.C, mod.iseas)
    phi_samps[:, day] = phi[0] + np.sqrt(var[0,0])*np.random.standard_t(mod.delVar*mod.n, size = [nsamps])
    return phi_samps

def sample_latent_factor_fxnl(day, L, m, C, iseas, seasPeriods, delVar, n, nsamps):
    phi_samps = np.zeros([nsamps, seasPeriods])
    phi, var = fourierToSeasonalFxnl(L, m, C, iseas)
    phi_samps[:, day] = phi[0] + np.sqrt(var[0,0])*np.random.standard_t(delVar*n, size = [nsamps])
    return phi_samps

def forecast_latent_factor(mod, k, today, period, sample = False, nsamps = 1):
    Gk = np.linalg.matrix_power(mod.G, k-1)
    a = Gk @ mod.a
    R = Gk @ mod.R @ Gk.T + (k-1)*mod.W
    
    if sample:
        return sample_latent_factor_fxnl((today + k - 1) % period, mod.L, a, R, mod.iseas, mod.seasPeriods[0], mod.delVar, mod.n, nsamps)
    else:
        return get_latent_factor_fxnl((today + k - 1) % period, mod.L, a, R, mod.iseas, mod.seasPeriods[0])
    
def forecast_path_latent_factor(mod, k, today, period, sample = False, nsamps = 1):
    phi_mu = [None for x in range(k)]
    phi_sigma = [None for x in range(k)]
    phi_psi = [[None for y in range(x)] for x in range(k)]
    phi_samps = np.zeros([k, nsamps])
    p = mod.seasPeriods[0]
    Rlist = [None for x in range(k)]
            
    for i in range(k):
        
        # Get the marginal a, R
        Gk = np.linalg.matrix_power(mod.G, i)
        a = Gk @ mod.a
        R = Gk @ mod.R @ Gk.T + (i)*mod.W
            
        Rlist[i] = R
        
        if sample:
            phi_samps[i, :] = sample_latent_factor_fxnl((today + i) % period, mod.L, a, R, mod.iseas, mod.seasPeriods[0], mod.delVar, mod.n, nsamps)
        else:
            phi_mu[i], phi_sigma[i] = get_latent_factor_fxnl((today + i) % period, mod.L, a, R, mod.iseas, mod.seasPeriods[0])
            
            # Find covariances with previous latent factor values
            for j in range(i):
                # Covariance matrix between the state vector at times j, i, i > j
                idx_i = (today + i) % period
                idx_j = (today + j) % period
                cov_ij = (np.linalg.matrix_power(mod.G, i-j) @ Rlist[j])[np.ix_(mod.iseas, mod.iseas)]
                phi_psi[i][j] = np.zeros([p, p])
                phi_psi[i][j][idx_i,idx_j] = (mod.L @ cov_ij @ mod.L.T)[idx_i, idx_j]
                            
    if sample:
        return phi_samps
    else:
        return phi_mu, phi_sigma, phi_psi
    
    