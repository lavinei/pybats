# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/14_latent_factor_fxns.ipynb (unless otherwise specified).

__all__ = ['update_lf_analytic', 'update_lf_analytic_dlm', 'forecast_marginal_lf_analytic', 'forecast_path_lf_copula',
           'update_lf_sample', 'update_lf_sample_forwardfilt', 'forecast_marginal_lf_sample', 'forecast_path_lf_sample',
           'forecast_joint_marginal_lf_copula', 'forecast_joint_marginal_lf_copula_dcmm', 'forecast_marginal_lf_dcmm',
           'forecast_path_lf_dcmm']

# Internal Cell
#exporti
import numpy as np

from .forecast import forecast_path_copula_sim, forecast_path_copula_density_MC, forecast_aR, \
    forecast_joint_copula_density_MC, forecast_joint_copula_sim
from .update import update_F
import multiprocessing
from functools import partial

# Internal Cell
def update_F_lf(mod, phi, F=None):
    if F is None:
        if mod.nlf > 0:
            mod.F[mod.ilf] = phi.reshape(mod.nlf, 1)
    else:
        if mod.nlf > 0:
            F[mod.ilf] = phi.reshape(mod.nlf, 1)
        return F

# Cell
def update_lf_analytic(mod, y = None, X = None, phi_mu = None, phi_sigma = None):


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
        update_F_lf(mod, phi_mu)

        # Mean and variance
        ft, qt = mod.get_mean_and_var_lf(mod.F, mod.a, mod.R, phi_mu, phi_sigma, mod.ilf)
        # if qt[0] < 0:
        #     print('correcting matrix')
        #     while qt<0:
        #         mod.R[np.diag_indices_from(mod.R)] += 0.001
        #         ft, qt = mod.get_mean_and_var_lf(mod.F, mod.a, mod.R, phi_mu, phi_sigma, mod.ilf)
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

# Cell
def update_lf_analytic_dlm(mod, y=None, X=None, phi_mu = None, phi_sigma = None):

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
        update_F_lf(mod, phi_mu)

        # Mean and variance
        ft, qt = mod.get_mean_and_var_lf(mod.F, mod.a, mod.R, phi_mu, phi_sigma, mod.ilf)
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
        # mod.C = (mod.R - qt * At @ At.T)

        # Get priors a, R for time t + 1 from the posteriors m, C
        mod.a = mod.G @ mod.m
        mod.R = mod.G @ mod.C @ mod.G.T
        mod.R = (mod.R + mod.R.T) / 2

        # Discount information
        mod.W = mod.get_W(X=X)
        mod.R = mod.R + mod.W
        mod.n = mod.delVar * mod.n

# Internal Cell
def get_mean_and_var_lf(self, F, a, R, phi_mu, phi_sigma, ilf):
    p = len(ilf)
    if p == 1:
        extra_var = a[ilf] ** 2 * phi_sigma + R[np.ix_(ilf, ilf)] * phi_sigma
    else:
        extra_var = a[ilf].T @ phi_sigma @ a[ilf] + np.trace(R[np.ix_(ilf, ilf)] @ phi_sigma)

    return F.T @ a, (F.T @ R @ F + extra_var) / self.rho

# Internal Cell
def get_mean_and_var_lf_dlm(F, a, R, phi_mu, phi_sigma, ilf, ct):
    p = len(ilf)
    if p == 1:
        extra_var = a[ilf] ** 2 * phi_sigma/ct * R[np.ix_(ilf, ilf)] * phi_sigma
    else:
        extra_var = a[ilf].T @ phi_sigma @ a[ilf]/ct + np.trace(R[np.ix_(ilf, ilf)] @ phi_sigma)

    return F.T @ a, F.T @ R @ F + extra_var

# Cell
def forecast_marginal_lf_analytic(mod, k, X = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False, state_mean_var = False):

    # Plug in the correct F values
    F = update_F(mod, X, F=mod.F.copy())

    # Put the mean of the latent factor phi_mu into the F vector
    F = update_F_lf(mod, phi_mu, F=F)

    a, R = forecast_aR(mod, k)

    # Mean and variance
    ft, qt = mod.get_mean_and_var_lf(F, a, R, phi_mu, phi_sigma, mod.ilf)

    if state_mean_var:
        return ft, qt

    # Choose conjugate prior, match mean and variance
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)

    if mean_only:
        return mod.get_mean(param1, param2)

    # Simulate from the forecast distribution
    return mod.simulate(param1, param2, nsamps)

# Cell
def forecast_path_lf_copula(mod, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = None, nsamps = 1, t_dist=False, y = None, nu=9, return_mu_cov=False):

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
        F = update_F_lf(mod, phi_mu[i], F=F)
        # if mod.nlf > 0:
        #     F[mod.ilf] = phi_mu[i].reshape(mod.nlf,1)

        Flist[i] = np.copy(F)

        # Find lambda mean and var
        ft, qt = mod.get_mean_and_var_lf(F, a, R, phi_mu[i], phi_sigma[i], mod.ilf)
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
                lambda_cov[j,i] = lambda_cov[i,j] = Flist[j].T @ cov_ij @ Flist[i] + \
                                                    alist[i][mod.ilf].T @ phi_psi[i-1][:,:,j] @ alist[j][mod.ilf] + \
                                                    np.trace(cov_ij[np.ix_(mod.ilf, mod.ilf)] @ phi_psi[i-1][:,:,j])

    if return_mu_cov:
        return lambda_mu, lambda_cov

    if y is not None:
        return forecast_path_copula_density_MC(mod, y, lambda_mu, lambda_cov, t_dist, nu, nsamps)
    else:
        return forecast_path_copula_sim(mod, k, lambda_mu, lambda_cov, nsamps, t_dist, nu)

# Cell
def update_lf_sample(mod, y = None, X = None, phi_samps = None, parallel=False):
    """
    DGLM update function with samples of a latent factor.

    $\phi_{samps}$ = Array of simulated values of a latent factor.
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
            f = partial(update_lf_sample_forwardfilt, mod, y, mod.F, mod.a, mod.R)
            p = multiprocessing.Pool(10)
            output = p.map(f, phi_samps)
            p.close()
        else:
            output = map(lambda p: update_lf_sample_forwardfilt(mod, y, mod.F, mod.a, mod.R, p), phi_samps)
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

# Cell
def update_lf_sample_forwardfilt(mod, y, F, a, R, phi):
    F = update_F_lf(mod, phi, F=F)
    # F[mod.ilf] = phi.reshape(-1,1)
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

# Cell
def forecast_marginal_lf_sample(mod, k, X = None, phi_samps = None, mean_only = False):

    # Plug in the correct F values
    F = update_F(mod, X, F=mod.F.copy())

    a, R = forecast_aR(mod, k)

    # Simulate from the forecast distribution
    return np.array(list(map(lambda p: lf_simulate_from_sample(mod, F, a, R, p), phi_samps))).reshape(-1)

# Internal Cell
def lf_simulate_from_sample(mod, F, a, R, phi):

    F = update_F_lf(mod, phi, F=F)
    # F[mod.ilf] = phi.reshape(-1,1)
    ft, qt = mod.get_mean_and_var(F, a, R)
    # get the conjugate prior parameters
    param1, param2 = mod.get_conjugate_params(ft, qt, mod.param1, mod.param2)
    # Update to the conjugate posterior after observing 'y'
    return mod.simulate(param1, param2, 1)

# Cell
def forecast_path_lf_sample(mod, k, X=None, phi_samps = None):

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
            F = update_F_lf(mod, phi_samps[n][i], F=F)
            # F[mod.ilf] = phi_samps[n][i].reshape(-1, 1)

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

# Cell
def forecast_joint_marginal_lf_copula(mod_list, k, X_list=None, phi_mu = None, phi_sigma = None,
                                      nsamps=1, y=None, t_dist=False, nu=9, return_cov=False):

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
        alist[i] = a[mod.ilf]

        # Plug in the correct F values
        if mod.nregn > 0:
            F = update_F(mod, X, F=mod.F.copy())
        else:
            F = mod.F.copy()

        # Put the mean of the latent factor phi_mu into the F vector
        F = update_F_lf(mod, phi_mu, F=F)

        Flist[i] = F

        # Find lambda mean and var
        ft, qt = mod.get_mean_and_var_lf(F, a, R, phi_mu, phi_sigma, mod.ilf)
        lambda_mu[i] = ft
        lambda_cov[i, i] = qt

        # Find covariances with lambda values from other models
        for j in range(i):
            # Covariance matrix between lambda from models i, j
            if phi_sigma.ndim == 0:
                lambda_cov[j, i] = lambda_cov[i, j] = np.squeeze(alist[i] * phi_sigma * alist[j])
            else:
                lambda_cov[j, i] = lambda_cov[i, j] = alist[i].T @ phi_sigma @ alist[j]

    if return_cov:
        return lambda_cov

    if y is not None:
        return forecast_joint_copula_density_MC(mod_list, y, lambda_mu, lambda_cov, t_dist, nu, nsamps)
    else:
        return forecast_joint_copula_sim(mod_list, lambda_mu, lambda_cov, nsamps, t_dist, nu)

# Cell
def forecast_joint_marginal_lf_copula_dcmm(dcmm_list, k, X_list=None, phi_mu = None, phi_sigma = None,
                                      nsamps=1, t_dist=False, nu=9, return_cov=False):

    bern_list = [mod.bern_mod for mod in dcmm_list]
    pois_list = [mod.pois_mod for mod in dcmm_list]

    mod_list = [*bern_list, *pois_list]

    p = len(mod_list)

    lambda_mu = np.zeros([p])
    lambda_cov = np.zeros([p, p])

    Flist = [None for x in range(p)]
    Rlist = [None for x in range(p)]
    alist = [None for x in range(p)]

    if X_list is None:
        X_list = [[] for i in range(p)]
    else:
        X_list = [*X_list, *X_list]


    for i, [X, mod] in enumerate(zip(X_list, mod_list)):

        # Evolve to the prior at time t + k
        a, R = forecast_aR(mod, k)

        Rlist[i] = R
        alist[i] = a[mod.ilf]

        # Plug in the correct F values
        if mod.nregn > 0:
            F = update_F(mod, X, F=mod.F.copy())
        else:
            F = mod.F.copy()

        # Put the mean of the latent factor phi_mu into the F vector
        F = update_F_lf(mod, phi_mu, F=F)

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

    samps = forecast_joint_copula_sim(mod_list, lambda_mu, lambda_cov, nsamps, t_dist, nu)

    bern_samps = samps[:,:len(bern_list)]
    pois_samps = samps[:, len(bern_list):]
    pois_samps += 1

    samps = bern_samps * pois_samps

    if return_cov:
        return np.cov(samps.T)

    return samps

# Cell
def forecast_marginal_lf_dcmm(mod, k, X=None, phi_mu=None, phi_sigma=None, nsamps=1, t_dist=False, nu=9, return_cov=False):

    mod_list = [mod.bern_mod, mod.pois_mod]
    lambda_mu = np.zeros(2)
    lambda_cov = np.zeros([2,2])
    a_lf_list=[]

    for i, mod in enumerate(mod_list):
        # Plug in the correct F values
        F = update_F(mod, X, F=mod.F.copy())
        # F = np.copy(mod.F)
        # if mod.nregn > 0:
        #     F[mod.iregn] = X.reshape(mod.nregn,1)

        # Put the mean of the latent factor phi_mu into the F vector
        F = update_F_lf(mod, phi_mu, F=F)
        # if mod.nlf > 0:
        #     F[mod.ilf] = phi_mu.reshape(mod.nlf,1)

        a, R = forecast_aR(mod, k)
        a_lf_list.append(a[mod.ilf])

        # Mean and variance
        ft, qt = mod.get_mean_and_var_lf(F, a, R, phi_mu, phi_sigma, mod.ilf)
        lambda_mu[i] = ft
        lambda_cov[i,i] = qt

    lambda_cov[0,1] = lambda_cov[1,0] = a_lf_list[0].T @ phi_sigma @ a_lf_list[1]

    samps = forecast_joint_copula_sim(mod_list, lambda_mu, lambda_cov, nsamps, t_dist, nu)

    bern_samps = samps[:, 0]
    pois_samps = samps[:, 1]
    pois_samps += 1

    samps = bern_samps * pois_samps

    if return_cov:
        return np.cov(samps.T)

    return samps

# Cell
def forecast_path_lf_dcmm(mod, k, X=None, phi_mu=None, phi_sigma=None, phi_psi=None, nsamps=1, t_dist=False, nu=9, return_cov=False):

    lambda_mu = np.zeros(k*2)
    lambda_cov = np.zeros([k*2, k*2])

    mucov_bern = forecast_path_lf_copula(mod.bern_mod, k, X, phi_mu, phi_sigma, phi_psi, return_mu_cov=True)
    mucov_pois = forecast_path_lf_copula(mod.pois_mod, k, X, phi_mu, phi_sigma, phi_psi, return_mu_cov=True)
    lambda_mu[:k] = mucov_bern[0]
    lambda_mu[k:] = mucov_pois[0]
    lambda_cov[:k,:k] = mucov_bern[1]
    lambda_cov[k:,k:] = mucov_pois[1]


    for i in range(k):
        a_bern, R_bern = forecast_aR(mod.bern_mod, i+1)
        for j in range(k):
            a_pois, R_pois = forecast_aR(mod.pois_mod, j+1)
            if i == j:
                cov = float(a_bern[mod.bern_mod.ilf].T @ phi_sigma[i] @ a_pois[mod.pois_mod.ilf])
            elif i > j:
                 cov = float(a_bern[mod.bern_mod.ilf].T @ phi_psi[i-1][j] @ a_pois[mod.pois_mod.ilf])
            elif j > i:
                cov = float(a_bern[mod.bern_mod.ilf].T @ phi_psi[j-1][i] @ a_pois[mod.pois_mod.ilf])
            lambda_cov[i, j + k] = lambda_cov[j + k, i] = cov

    mod_list = [*[mod.bern_mod]*k, *[mod.pois_mod]*k]
    samps = forecast_joint_copula_sim(mod_list, lambda_mu, lambda_cov, nsamps, t_dist, nu)

    bern_samps = samps[:, :k]
    pois_samps = samps[:, k:]
    pois_samps += 1

    samps = bern_samps * pois_samps

    if return_cov:
        return np.cov(samps.T)

    return samps