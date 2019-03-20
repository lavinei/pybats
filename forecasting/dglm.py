# These are for the general DGLM
import numpy as np
import scipy as sc
from .seasonal import seascomp, createFourierToSeasonalL
from .update import update, update_normaldlm, update_bindglm
from .forecast import forecast_marginal, forecast_path, forecast_path_approx, forecast_marginal_bindglm, forecast_path_normaldlm
from .multiscale import multiscale_forecast_marginal, multiscale_forecast_marginal_approx, multiscale_forecast_path_approx
from .multiscale import multiscale_update, multiscale_update_approx, multiscale_get_mean_and_var

# These are for the bernoulli and Poisson DGLMs
from scipy.special import digamma
from scipy.special import beta as beta_fxn
from scipy import optimize as opt
from functools import partial
from scipy import stats

class dglm:
    
    def __init__(self,
                 a0 = None, 
                 R0 = None,
                 nregn = 0,
                 ntrend = 0,
                 nmultiscale = 0,
                 seasPeriods = None,
                 seasHarmComponents = None,
                 deltrend = 1, delregn = 1,
                 delmultiscale = 1,
                 delhol = 1, delseas = 1):
        """
        a0, R0: time 0 prior means/covariance for the state vector
        nregn: Number of regression components
        nmultiscale: Number of regression components associated with latent factors (for multi-scale inference)
        ntrend: Number of trend components (local level = 1, local trend = 2)
        seasPeriods: list of periods of seasonal components
        seasHarmComponents: List of Lists of harmonic components associated with each seasonal period
        delX: Discount factors associated with the different parts of the DLM        
        """        
                
        # Setting up trend F, G matrices
        Ftrend = np.ones([ntrend]).reshape(-1,1)
        i = 0
        self.itrend = list(range(i, ntrend))
        i += ntrend
        if ntrend == 0:
            Gtrend = np.empty([0, 0])
        # Local level
        elif ntrend == 1: 
            Gtrend = np.identity(ntrend)
        # Locally linear
        elif ntrend == 2:
            Gtrend = np.array([[1, 1], [0, 1]])

        # Setting up regression F, G matrices
        if nregn == 0:
            Fregn = np.empty([0]).reshape(-1,1)
            Gregn = np.empty([0, 0])
        else:
            Gregn = np.identity(nregn)
            Fregn = np.ones([nregn]).reshape(-1,1)
            self.iregn = list(range(i, i + nregn))
            i += nregn
            
        # Setting up multiscale F, G matrices and replacing necessary functions
        if nmultiscale == 0:
            Fmultiscale = np.empty([0]).reshape(-1,1)
            Gmultiscale = np.empty([0, 0])
        else:
            Gmultiscale = np.identity(nmultiscale)
            Fmultiscale = np.ones([nmultiscale]).reshape(-1,1)
            self.imultiscale = list(range(i, i + nmultiscale))
            i += nmultiscale

        # Setting up seasonal F, G matrices
        if seasPeriods is None:
            Fseas = np.empty([0]).reshape(-1,1)
            Gseas = np.empty([0, 0])
            nseas = 0
        elif len(seasPeriods) == 1:
            Fseas, Gseas = seascomp(seasPeriods[0], seasHarmComponents[0])
            nseas = Gseas.shape[0]
            self.L = createFourierToSeasonalL(seasPeriods[0], seasHarmComponents[0], Fseas, Gseas)
            self.iseas = list(range(i, i + nseas))
            i += nseas
        elif len(seasPeriods) > 1:
            output = list(map(seascomp, seasPeriods, seasHarmComponents))
            Flist = [x[0] for x in output]
            Glist = [x[1] for x in output]
            self.L = list(map(createFourierToSeasonalL, seasPeriods, seasHarmComponents, Flist, Glist))
            nseas = 2*sum(map(len, seasHarmComponents))
            Fseas = np.zeros([nseas]).reshape(-1,1)
            Gseas = np.zeros([nseas, nseas])
            idx = 0
            self.iseas = list(range(i, i + nseas))
            i += nseas
            for Fs, Gs in output:
                idx2 = idx + Fs.shape[0]
                Fseas[idx:idx2,0] = Fs
                Gseas[idx:idx2, idx:idx2] = Gs
                idx = idx2


        # Combine the F and G components together
        F = np.vstack([Ftrend, Fregn, Fmultiscale, Fseas])
        G = sc.linalg.block_diag(Gtrend, Gregn, Gmultiscale, Gseas)

        # Set up discount matrix
        Discount = sc.linalg.block_diag(deltrend*np.ones([ntrend, ntrend]),
                                        delregn*np.ones([nregn, nregn]),
                                        delmultiscale*np.ones([nmultiscale, nmultiscale]),
                                        delseas*np.ones([nseas, nseas]))
        Discount[Discount == 0] = 1
        
        
        self.param1 = 2 # Random initial guess
        self.param2 = 2 # Random initial guess
        self.nregn = nregn
        self.ntrend = ntrend
        self.nmultiscale = nmultiscale
        self.seasPeriods = seasPeriods
        self.nseas = nseas
        self.F = F
        self.G = G
        self.Discount = Discount     
        self.a = a0.reshape(-1,1)
        self.R = R0
        self.W = self.get_W()
        self.t = 0
        
    def update(self, y = None, X = None):
        update(self, y, X)

    def forecast_marginal(self, k, X = None, nsamps = 1, mean_only = False):
        return forecast_marginal(self, k, X, nsamps, mean_only)

    def forecast_path(self, k, X = None, nsamps = 1):
        return forecast_path(self, k, X, nsamps)

    def forecast_path_approx(self, k, X = None, nsamps = 1, t_dist=False):
        return forecast_path_approx(self, k, X, nsamps, t_dist)

    def multiscale_update(self, y = None, X = None, phi_samps = None, parallel=False):
        multiscale_update(self, y, X, phi_samps, parallel)

    def multiscale_update_approx(self, y = None, X = None, phi_mu = None, phi_sigma = None):
        multiscale_update_approx(self, y, X, phi_mu, phi_sigma)

    def multiscale_forecast_marginal(self, k, X = None, phi_samps = None, mean_only = False):
        return multiscale_forecast_marginal(self, k, X, phi_samps, mean_only)

    def multiscale_forecast_marginal_approx(self, k, X = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False):
        return multiscale_forecast_marginal_approx(self, k, X, phi_mu, phi_sigma, nsamps, mean_only)

    def multiscale_forecast_path_approx(self, k, X = None, phi_mu = None, phi_sigma = None, phi_psi = None, nsamps = 1, **kwargs):
        return multiscale_forecast_path_approx(self, k, X, phi_mu, phi_sigma, phi_psi, nsamps, **kwargs)
        
    def get_mean_and_var(self, F, a, R):
        return F.T @ a, F.T @ R @ F

    def multiscale_get_mean_and_var(self, F, a, R, phi_mu, phi_sigma, imultiscale):
        return multiscale_get_mean_and_var(F, a, R, phi_mu, phi_sigma, imultiscale)
    
    def get_W(self):
        return self.R/self.Discount - self.R        

    def save_params(self):
        """
        This is a stub
        """
        
    def simulate(self, param1, param2, nsamps):
        """
        this is a stub to simulate from the forecast distribution
        """
        
    def get_conjugate_params(self, ft, qt, param1, param2):
        """
        This is a stub... will be different for each type of DGLM
        """
        return param1, param2
    
    def get_mean(self, alpha, beta):
        """
        This is a stub... will be different for each type of DGLM
        """
    
    def update_conjugate_params(self, y, param1, param2):
        """
        This is a stub... will be different for each type of DGLM
        """
        ft_star = qt_star = None
        return param1, param2, ft_star, qt_star


class bern_dglm(dglm):
    
    def trigamma(self, x):
        return sc.special.polygamma(x = x, n = 1)

    def beta_approx(self, x, ft, qt):
        x = x**2
        return np.array([digamma(x[0]) - digamma(x[1]) - ft, self.trigamma(x = x[0]) + self.trigamma(x = x[1]) - qt]).reshape(-1)
    
    def get_conjugate_params(self, ft, qt, alpha, beta):
        # Choose conjugate prior, beta, and match mean & variance
        sol = opt.root(partial(self.beta_approx, ft = ft, qt = qt), x0 = np.sqrt(np.array([alpha, beta])))
        return sol.x**2
    
    def update_conjugate_params(self, y, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        alpha = alpha + y
        beta = beta + 1 - y

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - digamma(beta)
        qt_star = self.trigamma(alpha) + self.trigamma(beta)
        
        return alpha, beta, ft_star, qt_star
    
    def simulate(self, alpha, beta, nsamps):
        p = np.random.beta(alpha, beta, [nsamps])
        return np.random.binomial(1, p, size=[nsamps])
    
    def simulate_from_sampling_model(self, p, nsamps):
        return np.random.binomial(1, p, [nsamps])
    
    def prior_inverse_cdf(self, cdf, alpha, beta):
        return stats.beta.ppf(cdf, alpha, beta)

    def sampling_density(self, y, p):
        return stats.binom.pmf(n = 1, p = p, k = y)

    # NEED TO CHECK THIS FUNCTION - BEFORE, WAS JUST THE LAST RETURN LINE, WHICH I THINK IS THE PMF, NOT CDF
    def marginal_cdf(self, y, alpha, beta):
        if y == 1:
            return 1
        elif y == 0:
            return beta_fxn(y + alpha, 1 - y + beta)/ beta_fxn(alpha, beta)

    def loglik(self, y, alpha, beta):
        return stats.bernoulli.logpmf(y, alpha / (alpha + beta))
    
    def get_mean(self, alpha, beta):
        return alpha / (alpha + beta)
    
    def get_prior_var(self, alpha, beta):
        return (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        
        
class pois_dglm(dglm):
    
    def __init__(self, *args, rho = 1, **kwargs):
        self.rho = rho
        super().__init__(*args, **kwargs)
    
    def get_mean_and_var(self, F, a, R):
        return F.T @ a, F.T @ R @ F  / self.rho

    def multiscale_get_mean_and_var(self, F, a, R, phi_mu, phi_sigma, imultiscale):
        p = len(imultiscale)
        if p == 1:
            extra_var = a[imultiscale] ** 2 * phi_sigma + a[imultiscale] * R[
                np.ix_(imultiscale, imultiscale)] * phi_sigma
        else:
            extra_var = a[imultiscale].T @ phi_sigma @ a[imultiscale] + np.trace(
                R[np.ix_(imultiscale, imultiscale)] @ phi_sigma)

        return F.T @ a, (F.T @ R @ F + extra_var) / self.rho
    
    def trigamma(self, x):
        return sc.special.polygamma(x = x, n = 1)

    def gamma_approx(self, x, ft, qt):
        x = x**2 
        return np.array([digamma(x[0]) - np.log(x[1]) - ft, self.trigamma(x = x[0]) - qt]).reshape(-1)
    
    def get_conjugate_params(self, ft, qt, alpha, beta):
        # Choose conjugate prior, gamma, and match mean & variance
        sol = opt.root(partial(self.gamma_approx, ft = ft, qt = qt), x0 = np.sqrt(np.array([alpha, beta])))
        return sol.x**2
    
    def update_conjugate_params(self, y, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        alpha = alpha + float(y)
        beta = beta + 1

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - np.log(beta)
        qt_star = self.trigamma(alpha)
        
        return alpha, beta, ft_star, qt_star
    
    def simulate(self, alpha, beta, nsamps):
        return np.random.negative_binomial(alpha, beta/(1 + beta), [nsamps])
    
    def simulate_from_sampling_model(self, rate, nsamps):
        return np.random.poisson(rate, [nsamps])
    
    def prior_inverse_cdf(self, cdf, alpha, beta):
        return stats.gamma.ppf(cdf, a = alpha, scale = 1/beta)

    def sampling_density(self, y, mu):
        return stats.poisson.pmf(mu = mu, k = y)

    def marginal_cdf(self, y, alpha, beta):
        return stats.nbinom.cdf(y, alpha, beta/(1 + beta))

    def loglik(self, y, alpha, beta):
        return stats.nbinom.logpmf(y, alpha, beta/(1 + beta))
    
    def get_mean(self, alpha, beta):
        return alpha / beta
    
    def get_prior_var(self, alpha, beta):
        return alpha / beta**2
        
        
class normal_dlm(dglm):
    
    def __init__(self, *args, n0 = 1, s0 = 1, delVar = 1, **kwargs):
        self.delVar = delVar # Discount factor for the variance - using a beta-gamma random walk
        self.n = n0 # Prior sample size for the variance
        self.s = s0 # Prior mean for the variance
        super().__init__(*args, **kwargs)
        
    def get_mean_and_var(self, F, a, R):
        return F.T @ a, F.T @ R @ F + self.s

    def multiscale_get_mean_and_var(self, F, a, R, phi_mu, phi_sigma, imultiscale):
        ft, qt = multiscale_get_mean_and_var(F, a, R, phi_mu, phi_sigma, imultiscale)
        qt = qt + self.s
        return ft, qt
    
    def get_conjugate_params(self, ft, qt, mean, var):
        return ft, qt
        
    def simulate(self, mean, var, nsamps):
        return mean + np.sqrt(var)*np.random.standard_t(self.n, size = [nsamps])

    def simulate_from_sampling_model(self, mean, var, nsamps):
        return np.random.normal(mean, var, nsamps)

    def update(self, y=None, X=None):
        update_normaldlm(self, y, X)


    def forecast_path(self, k, X = None, nsamps = 1):
        return forecast_path_normaldlm(self, k, X, nsamps)
    
    
class bin_dglm(dglm):
    
    def trigamma(self, x):
        return sc.special.polygamma(x = x, n = 1)

    def beta_approx(self, x, ft, qt):
        x = x**2
        return np.array([digamma(x[0]) - digamma(x[1]) - ft, self.trigamma(x = x[0]) + self.trigamma(x = x[1]) - qt]).reshape(-1)
    
    def get_conjugate_params(self, ft, qt, alpha, beta):
        # Choose conjugate prior, beta, and match mean & variance
        sol = opt.root(partial(self.beta_approx, ft = ft, qt = qt), x0 = np.sqrt(np.array([alpha, beta])))
        return sol.x**2
    
    def update_conjugate_params(self, n, y, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        alpha = alpha + y
        beta = beta + n - y

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - digamma(beta)
        qt_star = self.trigamma(alpha) + self.trigamma(beta)
        
        return alpha, beta, ft_star, qt_star
    
    def simulate(self, n, alpha, beta, nsamps):
        p = np.random.beta(alpha, beta, [nsamps])
        return np.random.binomial(n.astype(int), p, size=[nsamps])
    
    def simulate_from_sampling_model(self, n, p, nsamps):
        return np.random.binomial(n, p, [nsamps])
    
    def prior_inverse_cdf(self, cdf, alpha, beta):
        return stats.beta.ppf(cdf, alpha, beta)

    def marginal_cdf(self, y, n, alpha, beta):
        cdf = 0.0
        for i in range(y+1):
            cdf += sc.misc.comb(n, y) * beta_fxn(y + alpha, n - y + beta)/ beta_fxn(alpha, beta)
        return cdf
    
    def loglik(self, data, alpha, beta):
        n, y = data
        return stats.binom.logpmf(y, n, alpha / (alpha + beta))
    
    def get_mean(self, n, alpha, beta):
        return n*(alpha / (alpha + beta))
    
    def get_prior_var(self, alpha, beta):
        return (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

    def update(self, n = None, y = None, X = None):
        update_bindglm(self, n, y, X)

    def forecast_marginal(self, n, k, X = None, nsamps = 1, mean_only = False):
        return forecast_marginal_bindglm(self, n, k, X, nsamps, mean_only)