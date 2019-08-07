# These are for the general DGLM
import numpy as np
import scipy as sc
from collections.abc import Iterable
from .seasonal import seascomp, createFourierToSeasonalL
from .update import update, update_normaldlm, update_bindglm
from .forecast import forecast_marginal, forecast_path, forecast_path_approx,\
    forecast_marginal_bindglm, forecast_path_normaldlm, forecast_state_mean_and_var
from .multiscale import multiscale_forecast_marginal, multiscale_forecast_marginal_approx, \
    multiscale_forecast_path_approx
from .multiscale import multiscale_update, multiscale_update_approx, multiscale_get_mean_and_var, multiscale_update_normaldlm_approx
from .conjugates import trigamma, bern_conjugate_params, bin_conjugate_params, pois_conjugate_params

# These are for the bernoulli and Poisson DGLMs
from scipy.special import digamma
from scipy.special import beta as beta_fxn
from scipy import stats


class dglm:

    def __init__(self,
                 a0=None,
                 R0=None,
                 nregn=0,
                 ntrend=0,
                 nmultiscale=0,
                 nhol=0,
                 seasPeriods=[],
                 seasHarmComponents=[],
                 deltrend=1, delregn=1,
                 delmultiscale=1,
                 delhol=1, delseas=1,
                 interpolate=True,
                 adapt_discount=False,
                 adapt_factor=0.5,
                 discount_forecast=False):
        """
        :param a0: Prior mean vector
        :param R0: Prior covariance matrix
        :param nregn: Number of regression components
        :param ntrend: Number of trend components
        :param nmultiscale: Number of multiscale components
        :param seasPeriods: List of periods of seasonal components
        :param seasHarmComponents: List of harmonic components included for each period
        :param deltrend: Discount factor on trend components
        :param delregn: Discount factor on regression components
        :param delmultiscale: Discount factor on multiscale components
        :param delhol: Discount factor on holiday components (currently deprecated)
        :param delseas: Discount factor on seasonal components
        :param interpolate: Whether to use interpolation for conjugate parameters
        :param adapt_discount: What method of discount adaption. False = None, 'info' = information based,\
         'positive-regn' = only discount if regression information is available
        :param discount_forecast: Whether to use discounting into the future
        """

        # Setting up trend F, G matrices

        i = 0
        self.itrend = list(range(i, ntrend))
        i += ntrend
        if ntrend == 0:
            Gtrend = np.empty([0, 0])
            Ftrend = np.zeros([ntrend]).reshape(-1, 1)
        # Local level
        elif ntrend == 1:
            Gtrend = np.identity(ntrend)
            Ftrend = np.array([1]).reshape(-1, 1)
        # Locally linear
        elif ntrend == 2:
            Gtrend = np.array([[1, 1], [0, 1]])
            Ftrend = np.array([1, 0]).reshape(-1, 1)

        # Setting up regression F, G matrices
        if nregn == 0:
            Fregn = np.empty([0]).reshape(-1, 1)
            Gregn = np.empty([0, 0])
        else:
            Gregn = np.identity(nregn)
            Fregn = np.ones([nregn]).reshape(-1, 1)
            self.iregn = list(range(i, i + nregn))
            i += nregn

        # Setting up holiday F, G matrices (additional regression indicators components)
        if nhol == 0:
            Fhol = np.empty([0]).reshape(-1, 1)
            Ghol = np.empty([0, 0])
        else:
            Ghol = np.identity(nhol)
            Fhol = np.ones([nhol]).reshape(-1, 1)
            self.ihol = list(range(i, i + nhol))
            self.iregn.extend(self.ihol)  # Adding on to the self.iregn
            i += nhol

        # Setting up multiscale F, G matrices and replacing necessary functions
        if nmultiscale == 0:
            Fmultiscale = np.empty([0]).reshape(-1, 1)
            Gmultiscale = np.empty([0, 0])
        else:
            Gmultiscale = np.identity(nmultiscale)
            Fmultiscale = np.ones([nmultiscale]).reshape(-1, 1)
            self.imultiscale = list(range(i, i + nmultiscale))
            i += nmultiscale

        # Setting up seasonal F, G matrices
        if len(seasPeriods) == 0:
            Fseas = np.empty([0]).reshape(-1, 1)
            Gseas = np.empty([0, 0])
            nseas = 0
        # elif len(seasPeriods) == 1:
        #     Fseas, Gseas = seascomp(seasPeriods[0], seasHarmComponents[0])
        #     nseas = Gseas.shape[0]
        #     self.L = createFourierToSeasonalL(seasPeriods[0], seasHarmComponents[0], Fseas, Gseas)
        #     self.iseas = list(range(i, i + nseas))
        #     i += nseas
        # elif len(seasPeriods) > 1:
        else:
            output = list(map(seascomp, seasPeriods, seasHarmComponents))
            Flist = [x[0] for x in output]
            Glist = [x[1] for x in output]
            self.L = list(map(createFourierToSeasonalL, seasPeriods, seasHarmComponents, Flist, Glist))
            nseas = 2 * sum(map(len, seasHarmComponents))
            Fseas = np.zeros([nseas]).reshape(-1, 1)
            Gseas = np.zeros([nseas, nseas])
            idx = 0
            self.iseas = []
            for harmComponents in seasHarmComponents:
                self.iseas.append(list(range(i, i + 2 * len(harmComponents))))
                i += 2 * len(harmComponents)
            for Fs, Gs in output:
                idx2 = idx + Fs.shape[0]
                Fseas[idx:idx2, 0] = Fs.squeeze()
                Gseas[idx:idx2, idx:idx2] = Gs
                idx = idx2

        # Combine the F and G components together
        F = np.vstack([Ftrend, Fregn, Fhol, Fmultiscale, Fseas])
        G = sc.linalg.block_diag(Gtrend, Gregn, Ghol, Gmultiscale, Gseas)

        # store the discount info
        self.deltrend = deltrend
        self.delregn = delregn
        self.delhol = delhol
        self.delmultiscale = delmultiscale
        self.delseas = delseas

        self.ntrend = ntrend
        self.nregn = nregn + nhol  # Adding on nhol
        self.nregn_exhol = nregn
        self.nhol = nhol
        self.nmultiscale = nmultiscale
        self.nseas = nseas

        # Set up discount matrix
        self.discount_forecast = discount_forecast
        Discount = self.build_discount_matrix()

        self.param1 = 2  # Random initial guess
        self.param2 = 2  # Random initial guess

        self.seasPeriods = seasPeriods
        self.F = F
        self.G = G
        self.Discount = Discount
        self.a = a0.reshape(-1, 1)
        self.R = R0
        self.t = 0
        self.interpolate = interpolate
        self.adapt_discount = adapt_discount
        self.k = adapt_factor
        self.W = self.get_W()

    def build_discount_matrix(self, X=None):
        # build up discount factors while possibly taking special care to not discount when the "regn"
        # type factors are zero

        # do this all with matrix slicing which is much faster than the block diag
        p = np.sum([self.ntrend, self.nregn_exhol, self.nhol, self.nmultiscale, self.nseas])
        # start with no discounting
        component_discounts = np.ones([p, p])
        i = 0 # this will be the offset of the current block
        for discount_pair, n in zip([('std', self.deltrend), ('regn', self.delregn), ('regn', self.delhol),
                                     ('std', self.delmultiscale), ('std', self.delseas)],
                                    [self.ntrend, self.nregn_exhol, self.nhol, self.nmultiscale, self.nseas]):
            discount_type, discount = discount_pair
            if n > 0:
                if isinstance(discount, Iterable):
                    if len(discount) < n:
                        raise ValueError('Error: Length of discount factors must be 1 or match component length')
                    for j, disc in enumerate(discount[:n]):
                        # fill the diags one at a time
                        component_discounts[i+j, i+j] = disc
                else:
                    # fill the block with the constant
                    component_discounts[i:(i+n), i:(i+n)] = discount

                # overwrite with ones if doing the positive logic
                if X is not None and self.adapt_discount == 'positive_regn' and discount_type == 'regn':
                    # offset of the regression params
                    regn_i = 0
                    # look through the regression params and set that slice on the
                    # discount to 1 if 0
                    for j in range(n):
                        if X[regn_i] == 0:
                            # set all discounts to one (i offsets the block and j offsets the regn param)
                            component_discounts[i + j, :] = 1.
                            component_discounts[:, i + j] = 1.
                        regn_i += 1

                # move on to the next block
                i += n

        return component_discounts

    def update(self, y=None, X=None):
        update(self, y, X)

    def forecast_marginal(self, k, X=None, nsamps=1, mean_only=False, state_mean_var=False):
        return forecast_marginal(self, k, X, nsamps, mean_only, state_mean_var)

    def forecast_path(self, k, X=None, nsamps=1):
        return forecast_path(self, k, X, nsamps)

    def forecast_path_approx(self, k, X=None, nsamps=1, **kwargs):
        return forecast_path_approx(self, k, X, nsamps, **kwargs)

    def forecast_state_mean_and_var(self, k, X = None):
        return forecast_state_mean_and_var(self, k, X)

    def multiscale_update(self, y=None, X=None, phi_samps=None, parallel=False):
        multiscale_update(self, y, X, phi_samps, parallel)

    def multiscale_update_approx(self, y=None, X=None, phi_mu=None, phi_sigma=None):
        multiscale_update_approx(self, y, X, phi_mu, phi_sigma)

    def multiscale_forecast_marginal(self, k, X=None, phi_samps=None, mean_only=False):
        return multiscale_forecast_marginal(self, k, X, phi_samps, mean_only)

    def multiscale_forecast_marginal_approx(self, k, X=None, phi_mu=None, phi_sigma=None, nsamps=1, mean_only=False, state_mean_var=False):
        return multiscale_forecast_marginal_approx(self, k, X, phi_mu, phi_sigma, nsamps, mean_only, state_mean_var)

    def multiscale_forecast_path_approx(self, k, X=None, phi_mu=None, phi_sigma=None, phi_psi=None, nsamps=1, **kwargs):
        return multiscale_forecast_path_approx(self, k, X, phi_mu, phi_sigma, phi_psi, nsamps, **kwargs)

    def get_mean_and_var(self, F, a, R):
        mean, var = F.T @ a, F.T @ R @ F
        return np.ravel(mean)[0], np.ravel(var)[0]

    def multiscale_get_mean_and_var(self, F, a, R, phi_mu, phi_sigma, imultiscale):
        return multiscale_get_mean_and_var(F, a, R, phi_mu, phi_sigma, imultiscale)

    def get_W(self, X=None):
        if self.adapt_discount == 'info':
            info = np.abs(self.a.flatten() / np.sqrt(self.R.diagonal()))
            diag = self.Discount.diagonal()
            diag = np.round(diag + (1 - diag) * np.exp(-self.k * info), 5)
            Discount = np.ones(self.Discount.shape)
            np.fill_diagonal(Discount, diag)
        elif self.adapt_discount == 'positive_regn' and X is not None:
            Discount = self.build_discount_matrix(X)
        else:
            Discount = self.Discount
        return self.R / Discount - self.R

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

    def get_mean_and_var(self, F, a, R):
        ft, qt = F.T @ a, F.T @ R @ F
        ft, qt = ft.flatten()[0], qt.flatten()[0]
        return ft, qt

    def get_conjugate_params(self, ft, qt, alpha, beta):
        # print('beta', ft, qt, np.sqrt(qt))
        # Choose conjugate prior, beta, and match mean & variance
        return bern_conjugate_params(ft, qt, alpha, beta, interp=self.interpolate)

    def update_conjugate_params(self, y, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        alpha = alpha + y
        beta = beta + 1 - y

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - digamma(beta)
        qt_star = trigamma(alpha) + trigamma(beta)

        # constrain this thing from going to crazy places?
        ft_star = max(-8, min(ft_star, 8))
        qt_star = max(0.001 ** 2, min(qt_star, 4 ** 2))

        return alpha, beta, ft_star, qt_star

    def simulate(self, alpha, beta, nsamps):
        p = np.random.beta(alpha, beta, [nsamps])
        return np.random.binomial(1, p, size=[nsamps])

    def simulate_from_sampling_model(self, p, nsamps):
        return np.random.binomial(1, p, [nsamps])

    def simulate_from_prior(self, alpha, beta, nsamps):
        return stats.beta.rvs(a=alpha, b=beta, size=nsamps)

    def prior_inverse_cdf(self, cdf, alpha, beta):
        return stats.beta.ppf(cdf, alpha, beta)

    def sampling_density(self, y, p):
        return stats.binom.pmf(n=1, p=p, k=y)

    # NEED TO CHECK THIS FUNCTION - BEFORE, WAS JUST THE LAST RETURN LINE, WHICH I THINK IS THE PMF, NOT CDF
    def marginal_cdf(self, y, alpha, beta):
        if y == 1:
            return 1
        elif y == 0:
            return beta_fxn(y + alpha, 1 - y + beta) / beta_fxn(alpha, beta)

    def loglik(self, y, alpha, beta):
        return stats.bernoulli.logpmf(y, alpha / (alpha + beta))

    def get_mean(self, alpha, beta):
        return np.ravel(alpha / (alpha + beta))[0]

    def get_prior_var(self, alpha, beta):
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))


class pois_dglm(dglm):

    def __init__(self, *args, rho=1, **kwargs):
        self.rho = rho
        super().__init__(*args, **kwargs)

    def get_mean_and_var(self, F, a, R):
        return F.T @ a, F.T @ R @ F / self.rho

    def multiscale_get_mean_and_var(self, F, a, R, phi_mu, phi_sigma, imultiscale):
        p = len(imultiscale)
        if p == 1:
            extra_var = a[imultiscale] ** 2 * phi_sigma + a[imultiscale] * R[
                np.ix_(imultiscale, imultiscale)] * phi_sigma
        else:
            extra_var = a[imultiscale].T @ phi_sigma @ a[imultiscale] + np.trace(
                R[np.ix_(imultiscale, imultiscale)] @ phi_sigma)

        return F.T @ a, (F.T @ R @ F + extra_var) / self.rho

    def get_conjugate_params(self, ft, qt, alpha, beta):
        # Choose conjugate prior, gamma, and match mean & variance
        # print('gamma', ft, qt)
        return pois_conjugate_params(ft, qt, alpha, beta, interp=self.interpolate)

    def update_conjugate_params(self, y, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        alpha = alpha + float(y)
        beta = beta + 1

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - np.log(beta)
        qt_star = trigamma(alpha)

        # constrain this thing from going to crazy places?
        qt_star = max(0.001 ** 2, min(qt_star, 4 ** 2))

        return alpha, beta, ft_star, qt_star

    def simulate(self, alpha, beta, nsamps):
        return np.random.negative_binomial(alpha, beta / (1 + beta), [nsamps])

    def simulate_from_sampling_model(self, rate, nsamps):
        return np.random.poisson(rate, [nsamps])

    def simulate_from_prior(self, alpha, beta, nsamps):
        return stats.gamma.rvs(a=alpha, scale=1/beta, size=nsamps)

    def prior_inverse_cdf(self, cdf, alpha, beta):
        return stats.gamma.ppf(cdf, a=alpha, scale=1 / beta)

    def sampling_density(self, y, mu):
        return stats.poisson.pmf(mu=mu, k=y)

    def marginal_cdf(self, y, alpha, beta):
        return stats.nbinom.cdf(y, alpha, beta / (1 + beta))

    def marginal_inverse_cdf(self, cdf, alpha, beta):
        return stats.nbinom.ppf(cdf, alpha, beta / (1 + beta))

    def loglik(self, y, alpha, beta):
        return stats.nbinom.logpmf(y, alpha, beta / (1 + beta))

    def get_mean(self, alpha, beta):
        return np.ravel(alpha/beta)[0]

    def get_prior_var(self, alpha, beta):
        return alpha / beta ** 2


class normal_dlm(dglm):

    def __init__(self, *args, n0=1, s0=1, delVar=1, **kwargs):
        self.delVar = delVar  # Discount factor for the variance - using a beta-gamma random walk
        self.n = n0  # Prior sample size for the variance
        self.s = s0  # Prior mean for the variance
        super().__init__(*args, **kwargs)

    def get_mean_and_var(self, F, a, R):
        return F.T @ a, F.T @ R @ F + self.s

    def multiscale_get_mean_and_var(self, F, a, R, phi_mu, phi_sigma, imultiscale):
        ft, qt = multiscale_get_mean_and_var(F, a, R, phi_mu, phi_sigma, imultiscale)
        qt = qt + self.s
        return ft, qt

    def get_mean(self, ft, qt):
        return np.ravel(ft)[0]

    def get_conjugate_params(self, ft, qt, mean, var):
        return ft, qt

    def simulate(self, mean, var, nsamps):
        return mean + np.sqrt(var) * np.random.standard_t(self.n, size=[nsamps])

    def simulate_from_sampling_model(self, mean, var, nsamps):
        return np.random.normal(mean, var, nsamps)

    def update(self, y=None, X=None):
        update_normaldlm(self, y, X)

    def forecast_path(self, k, X=None, nsamps=1):
        return forecast_path_normaldlm(self, k, X, nsamps)

    def multiscale_update_approx(self, y=None, X=None, phi_mu=None, phi_sigma=None):
        multiscale_update_normaldlm_approx(self, y, X, phi_mu, phi_sigma)


class bin_dglm(dglm):

    def get_conjugate_params(self, ft, qt, alpha, beta):
        # Choose conjugate prior, beta, and match mean & variance
        return bin_conjugate_params(ft, qt, alpha, beta, interp=self.interpolate)

    def update_conjugate_params(self, n, y, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        alpha = alpha + y
        beta = beta + n - y

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - digamma(beta)
        qt_star = trigamma(alpha) + trigamma(beta)

        # constrain this thing from going to crazy places?
        ft_star = max(-8, min(ft_star, 8))
        qt_star = max(0.001 ** 2, min(qt_star, 4 ** 2))

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
        for i in range(y + 1):
            cdf += sc.misc.comb(n, y) * beta_fxn(y + alpha, n - y + beta) / beta_fxn(alpha, beta)
        return cdf

    def loglik(self, data, alpha, beta):
        n, y = data
        return stats.binom.logpmf(y, n, alpha / (alpha + beta))

    def get_mean(self, n, alpha, beta):
        return np.ravel(n * (alpha / (alpha + beta)))[0]

    def get_prior_var(self, alpha, beta):
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    def update(self, n=None, y=None, X=None):
        update_bindglm(self, n, y, X)

    def forecast_marginal(self, n, k, X=None, nsamps=1, mean_only=False):
        return forecast_marginal_bindglm(self, n, k, X, nsamps, mean_only)
