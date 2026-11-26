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
