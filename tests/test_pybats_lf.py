import numpy as np
import sys

sys.path.insert(0,'../')
from pybats.dglm import pois_dglm, bern_dglm, dlm, bin_dglm
from pybats.dcmm import dcmm

def test_update():
    a0 = np.array([1, 1, 1])
    R0 = np.eye(3)
    mod_n = dlm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_p = pois_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_b = bin_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_bern = bern_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_dcmm = dcmm(a0_bern=a0, R0_bern=R0, ntrend_bern=1, nregn_bern = 2, delregn_bern=.9,
                    a0_pois=a0, R0_pois=R0, ntrend_pois=1, nregn_pois=2, delregn_pois=.9)
    # New data:
    y = 5
    X = np.array([1,2])

    # Test the Poisson DGLM
    mod_p.update(y=y, X=X)
    ans = np.array([[0.59974735],
       [0.59974735],
       [0.1994947 ]])
    assert (np.equal(np.round(ans, 5), np.round(mod_p.a, 5)).all())

    ans = np.array([-0.16107008, 0.93214436])
    assert (np.equal(np.round(ans, 5), np.round(mod_p.R[0:2, 1], 5)).all())

    # Test the Bernoulli DGLM
    mod_bern.update(y=1, X=X)
    ans = np.array([[1.02626224],
                    [1.02626224],
                    [1.05252447]])
    assert (np.equal(np.round(ans, 5), np.round(mod_bern.a, 5)).all())

    ans = np.array([-1.00331466e-04,  1.11099963])
    assert (np.equal(np.round(ans, 5), np.round(mod_bern.R[0:2, 1], 5)).all())

    # Test the DCMM
    mod_dcmm.update(y=y+1, X=X)
    assert(np.equal(mod_dcmm.pois_mod.a, mod_p.a).all())
    assert (np.equal(mod_dcmm.pois_mod.R[0:2,1], mod_p.R[0:2,1]).all())
    assert (np.equal(mod_dcmm.bern_mod.a, mod_bern.a).all())
    assert (np.equal(mod_dcmm.bern_mod.R[0:2, 1], mod_bern.R[0:2, 1]).all())

def test_forecast_marginal():
    a0 = np.array([1, 1, 1])
    R0 = np.eye(3)
    mod_n = dlm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9, discount_forecast=True)
    mod_p = pois_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_b = bin_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_bern = bern_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_dcmm = dcmm(a0_bern=a0, R0_bern=R0, ntrend_bern=1, nregn_bern=2, delregn_bern=.9,
                    a0_pois=a0, R0_pois=R0, ntrend_pois=1, nregn_pois=2, delregn_pois=.9)

    # New data:
    X = np.array([1,2])

    # Test the Poisson DGLM
    m_p = mod_p.forecast_marginal(k = 5, X=X, mean_only=True)
    ans = [232.15794803433968]
    assert (np.equal(np.round(ans[0], 5), np.round(m_p, 5)).all())

    # Test the Bernoulli DGLM
    m_bern = mod_bern.forecast_marginal(k=5, X=X, mean_only=True)
    ans = [0.9344642829948064]
    assert (np.equal(np.round(ans[0], 5), np.round(m_bern, 5)).all())

    # Test the DCMM
    m_dcmm = mod_dcmm.forecast_marginal(k=5, X=X, mean_only=True)
    assert (np.equal(m_dcmm, m_bern*(m_p+1)).all())

def test_forecast_path_copula():
    a0 = np.array([1, 1, 1])
    R0 = np.eye(3)/10
    mod_p = pois_dglm(a0, R0, ntrend=2, nregn=1, deltrend=1, delregn=.9)
    mod_dcmm = dcmm(a0_bern=a0, R0_bern=R0, ntrend_bern=2, nregn_bern=1, delregn_bern=.9,
                    a0_pois=a0, R0_pois=R0, ntrend_pois=2, nregn_pois=1, delregn_pois=.9)

    # New data:
    X = np.array([[1], [0.5]])

    # Test the DCMM
    samps = mod_dcmm.forecast_path_copula(k=2, X=X, nsamps=10000)
    m_samp = samps.mean(axis=0)
    m_marg = np.array([mod_dcmm.forecast_marginal(k=1, X=X[0], mean_only=True),
                       mod_dcmm.forecast_marginal(k=2, X=X[1], mean_only=True)]).reshape(-1)
    assert (np.equal(np.round(m_samp - m_marg, 0), np.zeros(2)).all())


def test_update_lf_analytic():
    a0 = np.array([1, 1, 1])
    R0 = np.eye(3) / 10
    mod_p = pois_dglm(a0, R0, ntrend=1, nlf=2, deltrend=1, dellf=1)
    mod_bern = bern_dglm(a0, R0, ntrend=1, nlf=2, deltrend=1, dellf=1)
    mod_dcmm = dcmm(a0_bern=a0, R0_bern=R0, ntrend_bern=1, nlf_bern=2,
                    a0_pois=a0, R0_pois=R0, ntrend_pois=1, nlf_pois=2)

    # New signal / latent factor
    phi_mu = np.array([1, 2])
    phi_sigma = np.eye(2)

    # New observation
    y = 8

    # Check Pois update
    mod_p.update_lf_analytic(y = y, phi_mu=phi_mu, phi_sigma=phi_sigma)

    ans = [0.93213802, 0.86427603]
    assert (np.equal(np.round(ans, 5), np.round(mod_p.a[1:], 5).reshape(-1)).all())

    ans = np.array([-0.00341616,  0.09658384])
    assert (np.equal(np.round(ans, 5), np.round(mod_p.R[0:2, 1], 5)).all())

    # Check Bern update
    mod_bern.update_lf_analytic(y=1, phi_mu=phi_mu, phi_sigma=phi_sigma)

    ans = [1.00203656, 1.00407312]
    assert (np.equal(np.round(ans, 5), np.round(mod_bern.a[1:], 5).reshape(-1)).all())

    ans = np.array([-8.63740667e-06,  9.99913626e-02])
    assert (np.equal(np.round(ans, 5), np.round(mod_bern.R[0:2, 1], 5)).all())

    # Check DCMM
    mod_dcmm.update_lf_analytic(y=y+1, phi_mu=phi_mu, phi_sigma=phi_sigma)
    assert (np.equal(mod_dcmm.pois_mod.a, mod_p.a).all())
    assert (np.equal(mod_dcmm.pois_mod.R[0:2, 1], mod_p.R[0:2, 1]).all())
    assert (np.equal(mod_dcmm.bern_mod.a, mod_bern.a).all())
    assert (np.equal(mod_dcmm.bern_mod.R[0:2, 1], mod_bern.R[0:2, 1]).all())


def test_forecast_marginal_lf_analytic():
    a0 = np.array([1, 1, 1])
    R0 = np.eye(3) / 10
    mod_p = pois_dglm(a0, R0, ntrend=1, nlf=2, deltrend=1, delregn=.9)
    mod_bern = bern_dglm(a0, R0, ntrend=1, nlf=2, deltrend=1, dellf=1)
    mod_dcmm = dcmm(a0_bern=a0, R0_bern=R0, ntrend_bern=1, nlf_bern=2,
                    a0_pois=a0, R0_pois=R0, ntrend_pois=1, nlf_pois=2)

    # New latent factor
    phi_mu = np.array([1, 2])
    phi_sigma = np.array([[1,.5], [.5, 1]])

    # Test the Pois mod
    m = mod_p.forecast_marginal_lf_analytic(k=5, phi_mu=phi_mu, phi_sigma=phi_sigma,
                                               mean_only=True)
    ans_p = 157.7892898396817
    assert (np.equal(np.round(ans_p, 5), np.round(m, 5)).all())

    # Test the Bern mod
    m = mod_bern.forecast_marginal_lf_analytic(k=5, phi_mu=phi_mu, phi_sigma=phi_sigma,
                                               mean_only=True)
    ans_bern = 0.9524841419507198
    assert (np.equal(np.round(ans_bern, 5), np.round(m, 5)).all())

    # Test the DCMM
    m = mod_dcmm.forecast_marginal_lf_analytic(k=5, phi_mu=phi_mu, phi_sigma=phi_sigma,
                                               mean_only=True)
    ans = (ans_p+1)*ans_bern
    assert (np.equal(np.round(ans, 5), np.round(m, 5)).all())


def test_forecast_path_lf_copula():
    a0 = np.array([1, 1, 1])
    R0 = np.eye(3) / 10
    mod_p = pois_dglm(a0, R0, ntrend=1, nlf=2, deltrend=1, delregn=.9)
    mod_dcmm = dcmm(a0_bern=a0, R0_bern=R0, ntrend_bern=1, nlf_bern=2,
                    a0_pois=a0, R0_pois=R0, ntrend_pois=1, nlf_pois=2)

    # New latent factor
    phi_mu = [np.array([1, .5]), np.array([0, -1])]
    phi_sigma = [np.array([[.1,.05], [.05, .1]]), np.array([[.1,.05], [.05, .1]])]

    samps = mod_p.forecast_path_lf_copula(k=2, phi_mu=phi_mu, phi_sigma=phi_sigma, nsamps=10000)
    m_samp = samps.mean(axis=0)
    m_marg = np.array([mod_p.forecast_marginal_lf_analytic(k=1, phi_mu=phi_mu[0], phi_sigma=phi_sigma[0], mean_only=True),
                       mod_p.forecast_marginal_lf_analytic(k=2, phi_mu=phi_mu[1], phi_sigma=phi_sigma[1], mean_only=True)]).reshape(-1)
    assert (np.equal(np.round(m_samp - m_marg, 0), np.zeros(2)).all())

    # Test the DCMM
    samps = mod_dcmm.forecast_path_lf_copula(k=2, phi_mu=phi_mu, phi_sigma=phi_sigma, nsamps=10000)
    m_samp = samps.mean(axis=0)
    m_marg = np.array([mod_dcmm.forecast_marginal_lf_analytic(k=1, phi_mu=phi_mu[0], phi_sigma=phi_sigma[0], mean_only=True),
                       mod_dcmm.forecast_marginal_lf_analytic(k=2, phi_mu=phi_mu[1], phi_sigma=phi_sigma[1], mean_only=True)]).reshape(-1)
    assert (np.equal(np.round(m_samp - m_marg, 0), np.zeros(2)).all())