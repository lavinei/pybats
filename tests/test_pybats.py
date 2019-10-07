import numpy as np
import sys

sys.path.insert(0,'../')
from pybats.dglm import dlm, pois_dglm, bern_dglm, bin_dglm


def test_update():
    a0 = np.array([1, 1, 1])
    R0 = np.eye(3)
    mod_n = dlm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_p = pois_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_b = bin_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_bern = bern_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)

    # New data:
    y = 5
    X = np.array([1,2])

    # Test the normal DLM
    mod_n.update(y = y, X=X)
    ans = np.array([[1.14285714],
       [1.14285714],
       [1.28571429]])
    assert(np.equal(np.round(ans, 5), np.round(mod_n.a, 5)).all())

    ans = np.array([-0.08163265, 0.54421769])
    assert(np.equal(np.round(ans, 5), np.round(mod_n.R[0:2,1], 5)).all())

    # Test the Poisson DGLM
    mod_p.update(y=y, X=X)
    ans = np.array([[0.59974735],
       [0.59974735],
       [0.1994947 ]])
    assert (np.equal(np.round(ans, 5), np.round(mod_p.a, 5)).all())

    ans = np.array([-0.16107008, 0.93214436])
    assert (np.equal(np.round(ans, 5), np.round(mod_p.R[0:2, 1], 5)).all())

    # Test the Binomial DGLM
    mod_b.update(y=y, X=X, n=10)
    ans = np.array([[ 0.46543905],
       [ 0.46543905],
       [-0.0691219 ]])
    assert (np.equal(np.round(ans, 5), np.round(mod_b.a, 5)).all())

    ans = np.array([-0.15854342, 0.93495175])
    assert (np.equal(np.round(ans, 5), np.round(mod_b.R[0:2, 1], 5)).all())

    # Test the Bernoulli DGLM
    mod_bern.update(y=1, X=X)
    ans = np.array([[1.02626224],
                    [1.02626224],
                    [1.05252447]])
    assert (np.equal(np.round(ans, 5), np.round(mod_bern.a, 5)).all())

    ans = np.array([-1.00331466e-04,  1.11099963])
    assert (np.equal(np.round(ans, 5), np.round(mod_bern.R[0:2, 1], 5)).all())


def test_forecast_marginal():
    a0 = np.array([1, 1, 1])
    R0 = np.eye(3)
    mod_n = dlm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9, discount_forecast=True)
    mod_p = pois_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_b = bin_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)
    mod_bern = bern_dglm(a0, R0, ntrend=1, nregn=2, deltrend=1, delregn=.9)

    # New data:
    X = np.array([1,2])

    # Test the normal DLM
    m_n, v = mod_n.forecast_marginal(k = 5, X=X, state_mean_var=True)
    ans = [4., 9.222222222]
    assert(np.equal(np.round(ans[0], 5), np.round(m_n, 5)).all())
    assert(np.equal(np.round(ans[1], 5), np.round(v, 5)).all())

    # Test the Poisson DGLM
    m_p = mod_p.forecast_marginal(k = 5, X=X, mean_only=True)
    ans = [232.15794803433968]
    assert (np.equal(np.round(ans[0], 5), np.round(m_p, 5)).all())

    # Test the Binomial DGLM
    m_bin = mod_b.forecast_marginal(n = 10, k=5, X=X, mean_only=True)
    ans = [9.34464283]
    assert (np.equal(np.round(ans[0], 5), np.round(m_bin, 5)).all())

    # Test the Bernoulli DGLM
    m_bern = mod_bern.forecast_marginal(k=5, X=X, mean_only=True)
    ans = [0.9344642829948064]
    assert (np.equal(np.round(ans[0], 5), np.round(m_bern, 5)).all())


def test_forecast_path_copula():
    a0 = np.array([1, 1, 1])
    R0 = np.eye(3)/10
    mod_p = pois_dglm(a0, R0, ntrend=2, nregn=1, deltrend=1, delregn=.9)

    # New data:
    X = np.array([[1], [0.5]])

    # Test the Poisson DGLM
    samps_copula = mod_p.forecast_path_copula(k = 2, X=X, nsamps=10000)
    m_samp = samps_copula.mean(axis=0)
    m_marg = np.array([mod_p.forecast_marginal(k=1, X = X[0], mean_only=True),
              mod_p.forecast_marginal(k=2, X=X[1], mean_only=True)]).reshape(-1)
    assert (np.equal(np.round(m_samp - m_marg, 0), np.zeros(2)).all())

    samps_path = mod_p.forecast_path(k = 2, X=X, nsamps=2000, copula=False)
    m_samp = samps_path.mean(axis=0)
    assert (np.equal(np.round(m_samp - m_marg, 0), np.zeros(2)).all())
