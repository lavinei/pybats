from .dglm import bin_dglm
from .dcmm import dcmm
from functools import partial
import numpy as np

class dbcm:
    def __init__(self,
                 a0_bern = None, 
                 R0_bern = None,
                 nregn_bern = None,
                 ntrend_bern = 0,
                 nmultiscale_bern = 0,
                 seasPeriods_bern = [],
                 seasHarmComponents_bern = [],
                 deltrend_bern = 1, delregn_bern = 1,
                 delhol_bern = 1, delseas_bern = 1,
                 delmultiscale_bern = 1,
                 
                 a0_pois = None, 
                 R0_pois = None,
                 nregn_pois = None,
                 ntrend_pois = 0,
                 nmultiscale_pois = 0,
                 seasPeriods_pois = [],
                 seasHarmComponents_pois = [],
                 deltrend_pois = 1, delregn_pois = 1,
                 delhol_pois = 1, delseas_pois = 1,
                 delmultiscale_pois = 1,
                 rho = 1,
                
                 ncascade = 4,
                 a0_cascade = None, # List of length ncascade
                 R0_cascade = None, # List of length ncascade
                 nregn_cascade = None,
                 ntrend_cascade = 0,
                 nmultiscale_cascade = 0,
                 seasPeriods_cascade = [],
                 seasHarmComponents_cascade = [],
                 deltrend_cascade = 1, delregn_cascade = 1,
                 delhol_cascade = 1, delseas_cascade = 1,
                 delmultiscale_cascade = 1,

                 excess = []):
        """

        :param a0_bern: Prior mean vector for bernoulli DGLM
        :param R0_bern: Prior covariance matrix for bernoulli DGLM
        :param nregn_bern: Number of regression components in bernoulli DGLM
        :param ntrend_bern: Number of trend components in bernoulli DGLM
        :param nmultiscale_bern: Number of multiscale components in bernoulli DGLM
        :param seasPeriods_bern: List of periods of seasonal components in bernoulli DGLM
        :param seasHarmComponents_bern: List of harmonic components included for each period in bernoulli DGLM
        :param deltrend_bern: Discount factor on trend components in bernoulli DGLM
        :param delregn_bern: Discount factor on regression components in bernoulli DGLM
        :param delhol_bern: Discount factor on holiday component in bernoulli DGLM (currently deprecated)
        :param delseas_bern: Discount factor on seasonal components in bernoulli DGLM
        :param delmultiscale_bern: Discount factor on multiscale components in bernoulli DGLM
        :param a0_pois: Prior mean vector for poisson DGLM
        :param R0_pois: Prior covariance matrix for poisson DGLM
        :param nregn_pois: Number of regression components in poisson DGLM
        :param ntrend_pois: Number of trend components in poisson DGLM
        :param nmultiscale_pois: Number of multiscale components in poisson DGLM
        :param seasPeriods_pois: List of periods of seasonal components in poisson DGLM
        :param seasHarmComponents_pois: List of harmonic components included for each period in poisson DGLM
        :param deltrend_pois: Discount factor on trend components in poisson DGLM
        :param delregn_pois: Discount factor on regression components in poisson DGLM
        :param delhol_pois: Discount factor on holiday component in poisson DGLM (currently deprecated)
        :param delseas_pois: Discount factor on seasonal components in poisson DGLM
        :param delmultiscale_pois: Discount factor on multiscale components in poisson DGLM
        :param rho: Discount factor for random effects extension in poisson DGLM (smaller rho increases variance)
        :param ncascade: Number of cascade components in binary cascade
        :param a0_cascade: List of prior mean vectors for each binomial DGLM in cascade
        :param R0_cascade: List of prior covariance vectors for each binomial DGLM in cascade
        :param nregn_cascade: Number of regression components in poisson DGLM
        :param ntrend_cascade: Number of trend components in poisson DGLM
        :param nmultiscale_cascade: Number of multiscale components in poisson DGLM
        :param seasPeriods_cascade: List of periods of seasonal components in poisson DGLM
        :param seasHarmComponents_cascade: List of harmonic components included for each period in poisson DGLM
        :param deltrend_cascade: Discount factor on trend components in poisson DGLM
        :param delregn_cascade: Discount factor on regression components in poisson DGLM
        :param delhol_cascade: Discount factor on holiday component in poisson DGLM (currently deprecated)
        :param delseas_cascade: Discount factor on seasonal components in poisson DGLM
        :param delmultiscale_cascade: Discount factor on multiscale components in poisson DGLM
        :param excess: List of prior observed excess basket sizes >ncascade.
        """
        
        self.dcmm = dcmm(a0_bern, 
                 R0_bern,
                 nregn_bern,
                 ntrend_bern,
                 nmultiscale_bern,
                 seasPeriods_bern,
                 seasHarmComponents_bern,
                 deltrend_bern, delregn_bern,
                 delhol_bern, delseas_bern,
                 delmultiscale_bern,
                         
                 a0_pois, 
                 R0_pois,
                 nregn_pois,
                 ntrend_pois,
                 nmultiscale_pois,
                 seasPeriods_pois,
                 seasHarmComponents_pois,
                 deltrend_pois, delregn_pois,
                 delhol_pois, delseas_pois,
                 delmultiscale_pois,
                 rho = rho)
        
        self.ncascade = ncascade
        self.cascade = list(map(lambda a0, R0: bin_dglm(a0, R0, nregn_cascade, ntrend_cascade,
                                                       nmultiscale_cascade, seasPeriods_cascade,
                                                       seasHarmComponents_cascade, deltrend_cascade, delregn_cascade,
                                                       delhol_cascade, delseas_cascade, delmultiscale_cascade),
                                a0_cascade, R0_cascade))
        
        self.t = 0

        self.excess = excess

    def update_cascade(self, y_transaction = None, y_cascade = None, X_cascade = None):
        # Update the cascade of binomial DGLMs for basket sizes
        self.cascade[0].update(y_transaction, y_cascade[0], X_cascade)
        for i in range(1, self.ncascade):
            self.cascade[i].update(y_cascade[i - 1], y_cascade[i], X_cascade)


    def forecast_cascade(self, k, transaction_samps, X_cascade = None, nsamps = 1, mean_only=False):
        # forecast the sales from a cascade
        if mean_only:
            nsamps=1

        cascade_samps = np.zeros([self.ncascade, nsamps])
        cascade_samps[0, :] = self.cascade[0].forecast_marginal(transaction_samps, k, X_cascade, nsamps, mean_only)
        for i in range(1, self.ncascade):
            cascade_samps[i, :] = self.cascade[i].forecast_marginal(cascade_samps[i - 1, :], k, X_cascade, nsamps, mean_only)

        return cascade_samps

    def forecast_excess(self, max_cascade_samps, nsamps, mean_only=False):

        if mean_only:
            if len(self.excess) == 0:
                return np.array([(1)*max_cascade_samps]).reshape(1,1)
            else:
                return np.array([(np.mean(self.excess) - self.ncascade)*max_cascade_samps]).reshape(1,1)

        excess_samps = np.zeros([1, nsamps])
        sample = partial(np.random.choice, a=self.excess, replace=True)
        # If we have no prior data of any excess purchases, just assume the basket size
        # Is 1 greater than the last cascade we have in the model
        if len(self.excess) == 0:
            for idx in np.nonzero(max_cascade_samps)[0]:
                excess_samps[0, idx] = max_cascade_samps[idx] * 1
        else:
            for idx in np.nonzero(max_cascade_samps)[0]:
                excess_samps[0, idx] = np.sum(sample(size = max_cascade_samps[idx].astype(int))) - max_cascade_samps[idx] * self.ncascade

        return excess_samps

    # X is a list or tuple of length 3.
    # Data for the bernoulli DGLM, the Poisson DGLM, and then the cascade
    # Note we assume that all binomials in the cascade have the same regression components
    def update(self, y_transaction = None, X_transaction = None, y_cascade = None, X_cascade = None, excess = []):
        # Update the DCMM for transactions
        # We assume the bernoulli and poisson DGLMs have the same regression components
        self.dcmm.update(y_transaction, (X_transaction, X_transaction))
        self.update_cascade(y_transaction, y_cascade, X_cascade)
        # If there were any excess transactions, add that to the excess list
        self.excess.extend(excess)
        self.t += 1
        
    # Note we assume that the cascade is NOT multiscale, only the DCMM for transactions 
    def multiscale_update(self, y_transaction = None, X_transaction = None, y_cascade = None, X_cascade = None, phi_samps = None, excess = []):
        self.dcmm.multiscale_update(y_transaction, (X_transaction, X_transaction), (phi_samps, phi_samps))
        self.update_cascade(y_transaction, y_cascade, X_cascade)
        self.excess.extend(excess)
        self.t += 1
        
    def multiscale_update_approx(self, y_transaction = None, X_transaction = None, y_cascade = None, X_cascade = None, phi_mu = None, phi_sigma = None, excess = []):
        self.dcmm.multiscale_update_approx(y_transaction,
                                           (X_transaction, X_transaction),
                                           (phi_mu, phi_mu),
                                           (phi_sigma, phi_sigma))

        self.update_cascade(y_transaction, y_cascade, X_cascade)

        self.excess.extend(excess)
                
        self.t += 1
            
    def forecast_marginal(self, k, X_transaction = None, X_cascade = None, nsamps = 1, mean_only = False, return_separate = False):
        transaction_samps = self.dcmm.forecast_marginal(k, (X_transaction, X_transaction), nsamps, mean_only)
        cascade_samps = self.forecast_cascade(k, transaction_samps, X_cascade, nsamps, mean_only)
        excess_samps = self.forecast_excess(cascade_samps[self.ncascade-1,:], nsamps, mean_only)

        # Sometimes we may want to investigate the transaction, cascade, and excess samples separately
        if return_separate:
            return transaction_samps, cascade_samps, excess_samps

        samps = np.r_[transaction_samps.reshape(1, -1), cascade_samps, excess_samps.reshape(1, -1)]
        return np.sum(samps, axis = 0)
        
    def multiscale_forecast_marginal(self, k, X_transaction = None, X_cascade = None, phi_samps = None, nsamps = 1, mean_only = False, return_separate = False):
        transaction_samps = self.dcmm.multiscale_forecast_marginal(k, (X_transaction, X_transaction), (phi_samps, phi_samps), nsamps, mean_only)
        cascade_samps = self.forecast_cascade(k, transaction_samps, X_cascade, nsamps, mean_only)
        excess_samps = self.forecast_excess(cascade_samps[self.ncascade-1, :], nsamps, mean_only)

        # Sometimes we may want to investigate the transaction, cascade, and excess samples separately
        if return_separate:
            return transaction_samps, cascade_samps, excess_samps

        samps = np.r_[transaction_samps, cascade_samps, excess_samps]
        return np.sum(samps, axis=0)

    def multiscale_forecast_marginal_approx(self, k, X_transaction = None, X_cascade = None, phi_mu = None, phi_sigma = None, nsamps = 1, mean_only = False, return_separate=False):
        transaction_samps = self.dcmm.multiscale_forecast_marginal_approx(k, (X_transaction, X_transaction), (phi_mu, phi_mu), (phi_sigma, phi_sigma), nsamps, mean_only)
        cascade_samps = self.forecast_cascade(k, transaction_samps, X_cascade, nsamps, mean_only)
        excess_samps = self.forecast_excess(cascade_samps[self.ncascade-1, :], nsamps, mean_only)

        # Sometimes we may want to investigate the transaction, cascade, and excess samples separately
        if return_separate:
            return transaction_samps, cascade_samps, excess_samps

        samps = np.r_[transaction_samps, cascade_samps, excess_samps]
        return np.sum(samps, axis=0)
    
    def forecast_path(self, k, X_transaction = None, X_cascade = None, nsamps = 1, return_separate = False):
        transaction_samps = self.dcmm.forecast_path(k, (X_transaction, X_transaction), nsamps)
        cascade_samps = np.array(
            list(map(lambda h: self.forecast_cascade(h, transaction_samps[:, h], X_cascade[h], nsamps),
                     range(k)))).T
        excess_samps = np.array(list(map(lambda h: self.forecast_excess(cascade_samps[:, self.ncascade-1, h], nsamps),
                     range(k)))).T

        # Sometimes we may want to investigate the transaction, cascade, and excess samples separately
        if return_separate:
            return transaction_samps, cascade_samps, excess_samps


        samps = np.concatenate((transaction_samps[:, None, :], cascade_samps, excess_samps), axis=1)
        return np.sum(samps, axis=1)
    
    def forecast_path_approx(self, k, X_transaction = None, X_cascade = None, nsamps = 1, return_separate = False, **kwargs):
        transaction_samps = self.dcmm.forecast_path_approx(k, (X_transaction, X_transaction), nsamps, **kwargs)
        cascade_samps = np.array(
            list(map(lambda h: self.forecast_cascade(h, transaction_samps[:, h], X_cascade[h], nsamps),
                     range(k)))).T
        excess_samps = np.array(list(map(lambda h: self.forecast_excess(cascade_samps[:, self.ncascade-1, h], nsamps),
                                         range(k)))).T

        # Sometimes we may want to investigate the transaction, cascade, and excess samples separately
        if return_separate:
            return transaction_samps, cascade_samps, excess_samps

        samps = np.concatenate((transaction_samps[:, None, :], cascade_samps, excess_samps), axis=1)
        return np.sum(samps, axis=1)
    
    def multiscale_forecast_path_approx(self, k, X_transaction = None, X_cascade = None, phi_mu = None, phi_sigma = None, phi_psi = None, nsamps = 1, return_separate = False, **kwargs):
        transaction_samps = self.dcmm.multiscale_forecast_path_approx(k, (X_transaction, X_transaction), (phi_mu, phi_mu), (phi_sigma, phi_sigma), (phi_psi, phi_psi), nsamps, **kwargs)
        cascade_samps = np.array(
            list(map(lambda h: self.forecast_cascade(h, transaction_samps[:, h], X_cascade[h], nsamps),
                     range(k)))).T
        excess_samps = np.array(list(map(lambda h: self.forecast_excess(cascade_samps[:, self.ncascade-1, h], nsamps),
                                         range(k)))).T

        # Sometimes we may want to investigate the transaction, cascade, and excess samples separately
        if return_separate:
            return transaction_samps, cascade_samps, excess_samps

        samps = np.concatenate((transaction_samps[:, None, :], cascade_samps, excess_samps), axis=1)
        return np.sum(samps, axis=1)
        
        