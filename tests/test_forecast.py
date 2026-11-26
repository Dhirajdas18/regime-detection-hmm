import numpy as np
import pandas as pd

from rh import forecast as F
from rh import hmm


def test_ewma_uses_only_earlier_returns():
    x = pd.Series(np.random.default_rng(0).normal(0, 1, 100))
    a = F.ewma(x)
    y = x.copy()
    y.iloc[60:] *= 5
    assert np.allclose(a.iloc[:61], F.ewma(y).iloc[:61])  # day 60's forecast predates day 60


def test_one_state_hmm_forecast_is_the_past_normal():
    x = pd.Series(np.random.default_rng(1).normal(0.05, 1.2, 800),
                  index=pd.bdate_range("2000-01-03", periods=800))
    f = F.hmm_forecasts(x, K=1, first="2000-12-29")
    n = int(np.searchsorted(x.index, pd.Timestamp("2000-12-29"), side="right"))
    p = hmm.fit(x.to_numpy()[:n], 1, restarts=2)
    first = f.iloc[0]
    assert np.isclose(first["var"], p["sd"][0] ** 2)
    assert np.isclose(first["VaR"], p["mu"][0] - 2.3263479 * p["sd"][0], atol=1e-4)


def test_kupiec_is_zero_at_the_target_rate():
    b = np.zeros(1000, dtype=bool)
    b[:10] = True
    assert np.isclose(F.kupiec(b), 0) and F.kupiec(np.r_[b, np.ones(20, bool)]) > 3.84


def test_gjr_forecast_for_a_day_uses_only_earlier_days():
    x = pd.Series(np.random.default_rng(2).standard_t(5, 1600),
                  index=pd.bdate_range("2000-01-03", periods=1600))
    a = F.gjr(x, first="2003-12-31")
    y = x.copy()
    k = 1500
    y.iloc[k:] *= 4
    b = F.gjr(y, first="2003-12-31")
    d = x.index[k]
    assert np.allclose(a["var"].loc[:d], b["var"].loc[:d])  # day k's forecast predates day k
    assert not np.isclose(a["var"].iloc[-1], b["var"].iloc[-1])


def test_hmm_forecast_after_a_refit_refilters_the_past():
    from rh import hmm as H

    x = pd.Series(np.random.default_rng(3).normal(0, 1, 900),
                  index=pd.bdate_range("2000-01-03", periods=900))
    f = F.hmm_forecasts(x, K=2, first="2000-12-29")
    d = pd.Timestamp("2002-01-01")  # first day after the 2001 refit
    n = int(np.searchsorted(x.index, pd.Timestamp("2001-12-31"), side="right"))
    p = H.fit(x.to_numpy()[:n], 2, restarts=2)
    alpha, *_ = H.forward(x.to_numpy()[:n], p["pi"], p["A"], p["mu"], p["sd"])
    w = alpha[-1] @ p["A"]
    m = w @ p["mu"]
    assert np.isclose(f.at[d, "var"], w @ (p["sd"] ** 2 + p["mu"] ** 2) - m * m)
