# These are for the general DGLM
import numpy as np
import scipy as sc
import numba
from numba import jit, njit
from forecasting.multiscale import *
from forecasting.seasonal import *

# These are for the bernoulli and Poisson DGLMs
from scipy.special import digamma
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
        
    #def update(self, y =None, X = None):
    #if(self.nmultiscale > 0):
    #    update_multiscale_approx
    #else:
    #    update(self, y, X)
        
    def get_mean_and_var(self, F, a, R):
        return F.T @ a, F.T @ R @ F   
    
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
    
    def get_conjugate_params(self, ft, qt, mean, var):
        return ft, qt
        
    def simulate(self, mean, var, nsamps):
        return mean + np.sqrt(var)*np.random.standard_t(self.delVar*self.n, size = [nsamps])
    
    
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
    
    def update_conjugate_params(self, data, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        # Note that we pass in n and y as a tuple for data
        n, y = data
        alpha = alpha + y
        beta = beta + n - y

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - digamma(beta)
        qt_star = self.trigamma(alpha) + self.trigamma(beta)
        
        return alpha, beta, ft_star, qt_star
    
    def simulate(self, n, alpha, beta, nsamps):
        p = np.random.beta(alpha, beta, [nsamps])
        return np.random.binomial(n, p, size=[nsamps])
    
    def simulate_from_sampling_model(self, n, p, nsamps):
        return np.random.binomial(n, p, [nsamps])
    
    def prior_inverse_cdf(self, cdf, alpha, beta):
        return stats.beta.ppf(cdf, alpha, beta)
    
    def loglik(self, data, alpha, beta):
        n, y = data
        return stats.binom.logpmf(y, n, alpha / (alpha + beta))
    
    
    def get_mean(self, n, alpha, beta):
        return n*(alpha / (alpha + beta))
    
    def get_prior_var(self, alpha, beta):
        return (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        