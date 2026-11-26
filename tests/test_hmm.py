import numpy as np
import pytest

from rh import hmm


def _sim(T=4000, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    mu, sd = np.array([0.001, -0.002]), np.array([0.008, 0.025])
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = rng.choice(2, p=A[s[t - 1]])
    return mu[s] + sd[s] * rng.normal(size=T), s, A, mu, sd


def test_fit_recovers_a_two_state_model_calm_first():
    x, s, A, mu, sd = _sim()
    p = hmm.fit(x, 2)
    assert np.allclose(p["sd"], sd, rtol=0.1) and p["sd"][0] < p["sd"][1]
    assert np.allclose(np.diag(p["A"]), np.diag(A), atol=0.02)
    gamma, _, _ = hmm.smooth(x, p["pi"], p["A"], p["mu"], p["sd"])
    assert (gamma.argmax(1) == s).mean() > 0.9
    assert (hmm.viterbi(x, p["pi"], p["A"], p["mu"], p["sd"]) == s).mean() > 0.9


def test_filtered_uses_only_the_past():
    x, *_ = _sim(1000, 1)
    p = hmm.fit(x, 2, restarts=1)
    f_all, *_ = hmm.forward(x, p["pi"], p["A"], p["mu"], p["sd"])
    f_cut, *_ = hmm.forward(x[:600], p["pi"], p["A"], p["mu"], p["sd"])
    assert np.allclose(f_all[:600], f_cut)  # future days don't change the filtered past


def test_matches_hmmlearn_likelihood():
    hmmlearn = pytest.importorskip("hmmlearn.hmm")
    x, *_ = _sim(3000, 2)
    p = hmm.fit(x, 2)
    m = hmmlearn.GaussianHMM(2, covariance_type="diag", n_iter=500, tol=1e-8, random_state=0)
    m.fit(x[:, None])
    assert p["ll"] >= m.score(x[:, None]) - 1.0
    assert np.isclose(sorted(np.sqrt(m.covars_.ravel())), p["sd"], rtol=0.05).all()


def test_realtime_probabilities_never_use_later_days():
    import pandas as pd

    x, *_ = _sim(1500, 3)
    dates = pd.bdate_range("2000-01-03", periods=1500)
    a = hmm.realtime(x, dates, 2, first="2001-12-31", restarts=1)
    k = 1100  # change every return after day k
    y = x.copy()
    y[k:] = y[k:] * 3 + 0.5
    b = hmm.realtime(y, dates, 2, first="2001-12-31", restarts=1)
    start = int(np.searchsorted(dates, a["dates"][0]))
    n = k - start
    assert np.allclose(a["probs"][:n], b["probs"][:n])
    assert not np.allclose(a["probs"][n:], b["probs"][n:])


def test_logscore_is_the_predictive_density_of_each_day():
    import pandas as pd

    x, *_ = _sim(800, 5)
    dates = pd.bdate_range("2000-01-03", periods=800)
    rt = hmm.realtime(x, dates, 1, first="2000-12-29", restarts=1)
    # one state = a normal with the past mean and sd: the score is its log density
    n = int(np.searchsorted(dates, pd.Timestamp("2000-12-29"), side="right"))
    p = hmm.fit(x[:n], 1, restarts=1)
    z = (x[n] - p["mu"][0]) / p["sd"][0]
    assert np.isclose(rt["logscore"][0], -0.5 * z * z - np.log(p["sd"][0] * np.sqrt(2 * np.pi)))


def test_smooth_matches_brute_force_enumeration():
    import itertools

    x = np.array([0.3, -2.0, 0.1, 1.5, -0.2])
    pi, A = np.array([0.6, 0.4]), np.array([[0.9, 0.1], [0.2, 0.8]])
    mu, sd = np.array([0.1, -0.3]), np.array([0.8, 2.0])
    dens = lambda v, k: np.exp(-0.5 * ((v - mu[k]) / sd[k]) ** 2) / (sd[k] * np.sqrt(2 * np.pi))  # noqa: E731
    paths = list(itertools.product([0, 1], repeat=len(x)))
    def weight(p):
        steps = [A[p[t - 1], p[t]] * dens(x[t], p[t]) for t in range(1, len(x))]
        return pi[p[0]] * dens(x[0], p[0]) * np.prod(steps)

    w = np.array([weight(p) for p in paths])
    pw = list(zip(w, paths, strict=True))
    gamma, xi, ll = hmm.smooth(x, pi, A, mu, sd)
    assert np.isclose(ll, np.log(w.sum()))
    for t in range(len(x)):
        assert np.isclose(gamma[t, 1], sum(wi for wi, p in pw if p[t] == 1) / w.sum())
    pair = sum(wi for wi, p in pw for t in range(len(x) - 1) if p[t] == 0 and p[t + 1] == 1)
    assert np.isclose(xi[0, 1], pair / w.sum())
    best = paths[int(np.argmax(w))]
    assert list(hmm.viterbi(x, pi, A, mu, sd)) == list(best)
