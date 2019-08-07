# These are for the general DGLM
import numpy as np
import scipy as sc
from .forecast import forecast_path_approx_sim, forecast_path_approx_dens_MC, forecast_aR, \
    forecast_joint_approx_dens_MC, forecast_joint_approx_sim
from .update import update_F
import multiprocessing
from functools import partial

def update_F_multiscale(mod, phi, F=None):
    if F is None:
        if mod.nmultiscale > 0:
            mod.F[mod.imultiscale] = phi.reshape(mod.nmultiscale, 1)
    else:
        if mod.nmultiscale > 0:
            F[mod.imultiscale] = phi.reshape(mod.nmultiscale, 1)
        return F


def multiscale_update(mod, y = None, X = None, phi_samps = None, parallel=False):
    """
    phi_samps: array of samples of the latent factor vector
    """

    # If data is missing then skip discounting and updating, posterior = prior
    if y is None or np.isnan(y):
        mod.t += 1
        mod.m = mod.a
        mod.C = mod.R
        
        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T)/2
        
        mod.W = mod.get_W(X=X)
            
    else:

        update_F(mod, X)

        # Update m, C using a weighted average of the samples
        if parallel:
            f = partial(multiscale_update_with_samp, mod, y, mod.F, mod.a, mod.R)
            p = multiprocessing.Pool(10)
            output = p.map(f, phi_samps)
            p.close()
        else:
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
        mod.W = mod.get_W(X=X)
        mod.R = mod.R + mod.W
        
                
def multiscale_update_with_samp(mod, y, F, a, R, phi):
    F = update_F_multiscale(mod, phi, F=F)
    # F[mod.imultiscale] = phi.reshape(-1,1)
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
    
    return m, C, np.ravel(loglik)[0]


def multiscale_update_approx(mod, y = None, X = None, phi_mu = None, phi_sigma = None):
    """
    phi_mu: mean vector of the latent factor
    phi_sigma: variance matrix of the latent factor
        
    Implementing approximation: Assume the latent factor is independent of the state vector
    """

    # If data is missing then skip discounting and updating, posterior = prior
    if y is None or np.isnan(y):
        mod.t += 1
        mod.m = mod.a
        mod.C = mod.R
        
        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T)/2
        
        mod.W = mod.get_W(X=X)
            
    else:

        update_F(mod, X)

        # Put the mean of the latent factor phi_mu into the F vector
        update_F_multiscale(mod, phi_mu)

        # Mean and variance
        ft, qt = mod.multiscale_get_mean_and_var(mod.F, mod.a, mod.R, phi_mu, phi_sigma, mod.imultiscale)
        # if qt[0] < 0:
        #     print('correcting matrix')
        #     while qt<0:
        #         mod.R[np.diag_indices_from(mod.R)] += 0.001
        #         ft, qt = mod.multiscale_get_mean_and_var(mod.F, mod.a, mod.R, phi_mu, phi_sigma, mod.imultiscale)
        #     print(ft, qt)

        # Choose conjugate prior, match mean and variance
        # Initializing the optimization routine at 1,1 is important. At bad initializations, optimizer can shoot off to infinity.
        mod.param1, mod.param2 = mod.get_conjugate_params(ft, qt, 1, 1)
        if mod.param1 > 1E7:
            print('Numerical instabilities appearing in params of ' + str(type(mod)))

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
        mod.W = mod.get_W(X=X)
        mod.R = mod.R + mod.W


def multiscale_update_normaldlm_approx(mod, y=None, X=None, phi_mu = None, phi_sigma = None):

    # If data is missing then skip discounting and updating, posterior = prior
    if y is None or np.isnan(y):
        mod.t += 1
        mod.m = mod.a
        mod.C = mod.R

        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T) / 2

        mod.W = mod.get_W(X=X)

    else:
        update_F(mod, X)

        # Put the mean of the latent factor phi_mu into the F vector
        update_F_multiscale(mod, phi_mu)

        # Mean and variance
        ft, qt = mod.multiscale_get_mean_and_var(mod.F, mod.a, mod.R, phi_mu, phi_sigma, mod.imultiscale)
        mod.param1 = ft
        mod.param2 = qt

        # See time t observation y (which was passed into the update function)
        mod.t += 1

        # Update the  parameters:
        et = y - ft

        # Adaptive coefficient vector
        At = mod.R @ mod.F / qt

        # Volatility estimate ratio
        rt = (mod.n + et ** 2 / qt) / (mod.n + 1)

        # Kalman filter update
        mod.n = mod.n + 1
        mod.s = mod.s * rt
        mod.m = mod.a + At * et
        mod.C = rt * (mod.R - qt * At @ At.T)

        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T) / 2

        # Discount information
        mod.W = mod.get_W(X=X)
        mod.R = mod.R + mod.W
        mod.n = mod.delVar * mod.n
        
        
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
    F = update_F(mod, X, F=mod.F.copy())

    a, R = forecast_aR(mod, k)

    # Simulate from the forecast distribution
    return np.array(list(map(lambda p: multiscale_sim_with_samp(mod, F, a, R, p), phi_samps))).reshape(-1)
        

def multiscale_sim_with_samp(mod, F, a, R, phi):
    """
    Simulate 'y' values using a single sample of the latent factor phi
    """
    F = update_F_multiscale(mod, phi, F=F)
    # F[mod.imultiscale] = phi.reshape(-1,1)
    ft, qt = mod.get_mean_and_var(F, a, R)
    # get the conjugate prior parameters
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)
    # Update to the conjugate posterior after observing 'y'    
    return mod.simulate(param1, param2, 1)


def multiscale_forecast_path(mod, k, X=None, phi_samps = None):
    """
    Forecast function for the k-step path with samples of the latent factor phi
    k: steps ahead to forecast
    X: array with k rows for the future regression components
    phi_samps: An nsamps length list of k-length lists with samples of the latent factor
    """

    nsamps = len(phi_samps)
    samps = np.zeros([nsamps, k])

    F = np.copy(mod.F)

    for n in range(nsamps):
        param1 = mod.param1
        param2 = mod.param2

        a = np.copy(mod.a)
        R = np.copy(mod.R)

        for i in range(k):

            # Plug in X values
            if mod.nregn > 0:
                F = update_F(mod, X[i, :], F=F)
            # if mod.nregn > 0:
            #     F[mod.iregn] = X[i, :].reshape(mod.nregn, 1)

            # Plug in phi sample
            F = update_F_multiscale(mod, phi_samps[n][i], F=F)
            # F[mod.imultiscale] = phi_samps[n][i].reshape(-1, 1)

            # Get mean and variance
            ft, qt = mod.get_mean_and_var(F, a, R)

            # Choose conjugate prior, match mean and variance
            param1, param2 = mod.get_conjugate_params(ft, qt, param1, param2)

            # Simulate next observation
            samps[n, i] = mod.simulate(param1, param2, nsamps=1)

            # Update based on that observation
            param1, param2, ft_star, qt_star = mod.update_conjugate_params(samps[n, i], param1, param2)

            # Kalman filter update on the state vector (using Linear Bayes approximation)
            m = a + R @ F * (ft_star - ft) / qt
            C = R - R @ F @ F.T @ R * (1 - qt_star / qt) / qt

            # Get priors a, R for the next time step
            a = mod.G @ m
            R = mod.G @ C @ mod.G.T
            R = (R + R.T) / 2

            # Discount information
            if mod.discount_forecast:
                R = R + mod.W

    return samps


def multiscale_forecast_marginal_approx(mod, k, X = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False, state_mean_var = False):
    """
    Forecast function k steps ahead (marginal)
    """
    # Plug in the correct F values
    F = update_F(mod, X, F=mod.F.copy())
    # F = np.copy(mod.F)
    # if mod.nregn > 0:
    #     F[mod.iregn] = X.reshape(mod.nregn,1)
            
    # Put the mean of the latent factor phi_mu into the F vector
    F = update_F_multiscale(mod, phi_mu, F=F)
    # if mod.nmultiscale > 0:
    #     F[mod.imultiscale] = phi_mu.reshape(mod.nmultiscale,1)
        
    a, R = forecast_aR(mod, k)
            
    # Mean and variance
    ft, qt = mod.multiscale_get_mean_and_var(F, a, R, phi_mu, phi_sigma, mod.imultiscale)

    if state_mean_var:
        return ft, qt
        
    # Choose conjugate prior, match mean and variance
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)
    
    if mean_only:
        return mod.get_mean(param1, param2)
        
    # Simulate from the forecast distribution
    return mod.simulate(param1, param2, nsamps)


def multiscale_forecast_state_mean_and_var(mod, k, X = None, phi_mu = None, phi_sigma = None):
    """
       Forecast function k steps ahead (marginal)
       """
    # Plug in the correct F values
    F = update_F(mod, X, F=mod.F.copy())
    # if mod.nregn > 0:
    #     F = np.copy(mod.F)
    #     F[mod.iregn] = X.reshape(mod.nregn, 1)

    # Put the mean of the latent factor phi_mu into the F vector
    F = update_F_multiscale(mod, phi_mu, F=F)
    # if mod.nmultiscale > 0:
    #     F[mod.imultiscale] = phi_mu.reshape(mod.nmultiscale, 1)

    a, R = forecast_aR(mod, k)

    # Mean and variance
    ft, qt = mod.multiscale_get_mean_and_var(F, a, R, phi_mu, phi_sigma, mod.imultiscale)


def multiscale_forecast_path_approx(mod, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = None, nsamps = 1, t_dist=False, y = None, nu=9):
    """

    :param mod: Model of class DGLM
    :param k: steps ahead to forecast
    :param X: array with k rows for the future regression components
    :param phi_mu: length k list of mean vectors of the latent factors
    :param phi_sigma: length k list of variance matrices of the latent factors
    :param phi_psi: length k list of covariance matrices of phi_t+k and phi_t+j. Each element is another list, of length k.
    :param nsamps: Number of samples to draw from forecast distribution
    :param t_dist: Use t-copula? If false, Gaussian is assumed.
    :param y: Future path of observations y. If provided, output will be the forecast density of y.
    :param nu: Degrees of freedom for t-copula.
    :return:
    """
    
    lambda_mu = np.zeros([k])
    lambda_cov = np.zeros([k, k])
    
    F = np.copy(mod.F)
                        
    Flist = [None for x in range(k)]
    Rlist = [None for x in range(k)]
    alist = [None for x in range(k)]
    
    for i in range(k):

        # Get the marginal a, R
        a, R = forecast_aR(mod, i+1)
            
        alist[i] = a
        Rlist[i] = R

        # Plug in the correct F values
        if mod.nregn > 0:
            F = update_F(mod, X[i,:], F=F)
        # if mod.nregn > 0:
        #     F[mod.iregn] = X[i,:].reshape(mod.nregn,1)
            
        # Put the mean of the latent factor phi_mu into the F vector
        F = update_F_multiscale(mod, phi_mu[i], F=F)
        # if mod.nmultiscale > 0:
        #     F[mod.imultiscale] = phi_mu[i].reshape(mod.nmultiscale,1)
            
        Flist[i] = np.copy(F)
            
        # Find lambda mean and var
        ft, qt = mod.multiscale_get_mean_and_var(F, a, R, phi_mu[i], phi_sigma[i], mod.imultiscale)
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

    if y is not None:
        return forecast_path_approx_dens_MC(mod, y, lambda_mu, lambda_cov, t_dist, nu, nsamps)
    else:
        return forecast_path_approx_sim(mod, k, lambda_mu, lambda_cov, nsamps, t_dist, nu)

def multiscale_forecast_marginal_joint_approx(mod_list, k, X_list=None, phi_mu = None, phi_sigma = None, phi_psi = None,
                                       nsamps=1, y=None, t_dist=False, nu=9):
    """
    Forecast function for multiple models, marginally k-steps ahead
    """

    p = len(mod_list)

    lambda_mu = np.zeros([p])
    lambda_cov = np.zeros([p, p])

    Flist = [None for x in range(p)]
    Rlist = [None for x in range(p)]
    alist = [None for x in range(p)]

    if X_list is None:
        X_list = [[] for i in range(p)]

    for i, [X, mod] in enumerate(zip(X_list, mod_list)):

        # Evolve to the prior at time t + k
        a, R = forecast_aR(mod, k)

        Rlist[i] = R
        alist[i] = a[mod.imultiscale]

        # Plug in the correct F values
        F = update_F(mod, X[i, :], F=mod.F.copy())
        # F = np.copy(mod.F)
        # if mod.nregn > 0:
        #     F[mod.iregn] = X[i, :].reshape(mod.nregn, 1)

        # Put the mean of the latent factor phi_mu into the F vector
        F = update_F_multiscale(mod, phi_mu, F=F)
        # if mod.nmultiscale > 0:
        #     F[mod.imultiscale] = phi_mu.reshape(mod.nmultiscale,1)

        Flist[i] = F

        # Find lambda mean and var
        ft, qt = mod.get_mean_and_var(F, a, R)
        lambda_mu[i] = ft
        lambda_cov[i, i] = qt

        # Find covariances with lambda values from other models
        for j in range(i):
            # Covariance matrix between lambda from models i, j
            if phi_sigma.ndim == 0:
                lambda_cov[j, i] = lambda_cov[i, j] = np.squeeze(alist[i] * phi_sigma * alist[j])
            else:
                lambda_cov[j, i] = lambda_cov[i, j] = alist[i].T @ phi_sigma @ alist[j]

    if y is not None:
        return forecast_joint_approx_dens_MC(mod_list, y, lambda_mu, lambda_cov, t_dist, nu, nsamps)
    else:
        return forecast_joint_approx_sim(mod_list, lambda_mu, lambda_cov, nsamps, t_dist, nu)

################ FUNCTIONS TO EXTRACT MULTISCALE SIGNAL... E.G. HOLIDAY REGRESSION COEFFICIENTS ###############


def forecast_holiday_effect(mod, X, k):
    a, R = forecast_aR(mod, k)

    mean = X.T @ a[mod.ihol]
    var = X.T @ R[np.ix_(mod.ihol, mod.ihol)] @ X
    return mean, var


def combine_multiscale(*signals):
    phi_mu_prior = []
    phi_sigma_prior = []
    phi_mu_post = []
    phi_sigma_post = []
    num_signals = len(signals)
    k = len(signals[0][0][0]) # Forecast length
    prior_len = len(signals[0][0])
    post_len = len(signals[0][2])

    for i in range(post_len):
        mean = np.concatenate([signals[n][2][i] for n in range(num_signals)])
        var = sc.linalg.block_diag(*[signals[n][3][i] for n in range(num_signals)])
        phi_mu_post.append(mean)
        phi_sigma_post.append(var)

    for i in range(prior_len):
        mean = [np.concatenate([signals[n][0][i][j] if isinstance(signals[n][0][i][j], np.ndarray) else
                                np.array(signals[n][0][i][j]).reshape(-1) for n in range(num_signals)]) for j in range(k)]
        # mean = [np.concatenate([signals[n][0][i][j] for n in range(num_signals)]) for j in range(k)]
        var = [sc.linalg.block_diag(*[signals[n][1][i][j] for n in range(num_signals)]) for j in range(k)]
        phi_mu_prior.append(mean)
        phi_sigma_prior.append(var)

    return phi_mu_prior, phi_sigma_prior, phi_mu_post, phi_sigma_post