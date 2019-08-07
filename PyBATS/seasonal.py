import numpy as np

def seascomp(period, harmComponents):
    p = len(harmComponents)
    n = 2*p
    F = np.zeros([n, 1])
    F[0:n:2] = 1
    G = np.zeros([n, n])

    for j in range(p):
        c = np.cos(2*np.pi*harmComponents[j]/period)
        s = np.sin(2*np.pi*harmComponents[j]/period)
        idx = 2*j
        G[idx:(idx+2), idx:(idx+2)] = np.array([[c, s],[-s, c]])

    return [F, G]

def createFourierToSeasonalL(period, harmComponents, Fseas, Gseas):
    p = len(harmComponents)
    L = np.zeros([period, 2*p])
    L[0,:] = Fseas.reshape(-1)
    for i in range(1, period):
        L[i,:] = L[i-1,:] @ Gseas
        
    return L


def fourierToSeasonal(mod):
    phi = mod.L @ mod.m[mod.iseas]
    var = mod.L @ mod.C[np.ix_(mod.iseas, mod.iseas)] @ mod.L.T
    return phi, var

def fourierToSeasonalFxnl(L, m, C, iseas):
    phi = L @ m[iseas]
    var = L @ C[np.ix_(iseas, iseas)] @ L.T
    return phi, var

################# FUNCTIONS FOR EXTRACTING SEASONAL COMPONENTS (FOR MULTISCALE INFERENCE) ##############

def get_seasonal_effect_fxnl(L, m, C, iseas):
    phi, var = fourierToSeasonalFxnl(L, m, C, iseas)
    return phi[0], var[0, 0]

def sample_seasonal_effect_fxnl(L, m, C, iseas, seasPeriods, delVar, n, nsamps):
    phi_samps = np.zeros([nsamps])
    phi, var = fourierToSeasonalFxnl(L, m, C, iseas)
    phi_samps[:] = phi[0] + np.sqrt(var[0,0])*np.random.standard_t(delVar*n, size = [nsamps])
    return phi_samps

def forecast_weekly_seasonal_factor(mod, k, sample = False, nsamps = 1):
    Gk = np.linalg.matrix_power(mod.G, k-1)
    a = Gk @ mod.a
    R = Gk @ mod.R @ Gk.T + (k-1)*mod.W

    idx = np.where(np.array(mod.seasPeriods) == 7)[0][0]

    if sample:
        return sample_seasonal_effect_fxnl(mod.L[idx], a, R, mod.iseas[idx], mod.seasPeriods[idx], mod.delVar, mod.n, nsamps)
    else:
        return get_seasonal_effect_fxnl(mod.L[idx], a, R, mod.iseas[idx])


def forecast_path_seasonal_factor(mod, k, today, period, sample = False, nsamps = 1):
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
            phi_samps[i, :] = sample_seasonal_effect_fxnl_old((today + i) % period, mod.L, a, R, mod.iseas, mod.seasPeriods[0], mod.delVar, mod.n, nsamps)
        else:
            phi_mu[i], phi_sigma[i] = get_latent_factor_fxnl_old((today + i) % period, mod.L, a, R, mod.iseas, mod.seasPeriods[0])

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


def sample_seasonal_effect_fxnl_old(day, L, m, C, iseas, seasPeriods, delVar, n, nsamps):
    phi_samps = np.zeros([nsamps, seasPeriods])
    phi, var = fourierToSeasonalFxnl(L, m, C, iseas)
    phi_samps[:, day] = phi[0] + np.sqrt(var[0,0])*np.random.standard_t(delVar*n, size = [nsamps])
    return phi_samps

def get_latent_factor_fxnl_old(day, L, m, C, iseas, seasPeriods):
    phi_mu = np.zeros([seasPeriods, 1])
    phi_sigma = np.zeros([seasPeriods, seasPeriods])
    phi, var = fourierToSeasonalFxnl(L, m, C, iseas)
    phi_mu[day] = phi[0]
    phi_sigma[day, day] = var[0, 0]
    return phi_mu, phi_sigma


def forecast_weekly_seasonal_factor_old(mod, k, today, period, sample = False, nsamps = 1):
    Gk = np.linalg.matrix_power(mod.G, k-1)
    a = Gk @ mod.a
    R = Gk @ mod.R @ Gk.T + (k-1)*mod.W

    if sample:
        return sample_seasonal_effect_fxnl_old((today + k - 1) % period, mod.L[0], a, R, mod.iseas[0], mod.seasPeriods[0], mod.delVar, mod.n, nsamps)
    else:
        return get_latent_factor_fxnl_old((today + k - 1) % period, mod.L[0], a, R, mod.iseas[0], mod.seasPeriods[0])

def get_seasonal_effect_old(mod, day):
    phi_mu = np.zeros([mod.seasPeriods[0], 1])
    phi_sigma = np.zeros([mod.seasPeriods[0], mod.seasPeriods[0]])
    phi, var = fourierToSeasonal(mod)
    phi_mu[day] = phi[0]
    phi_sigma[day, day] = var[0, 0]
    return phi_mu, phi_sigma

def sample_seasonal_effect_old(mod, day, nsamps):
    phi_samps = np.zeros([nsamps, mod.seasPeriods[0]])
    phi, var = fourierToSeasonalFxnl(mod.L[0], mod.m, mod.C, mod.iseas[0])
    phi_samps[:, day] = phi[0] + np.sqrt(var[0,0])*np.random.standard_t(mod.delVar*mod.n, size = [nsamps])
    return phi_samps