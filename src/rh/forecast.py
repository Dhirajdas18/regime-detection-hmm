"""One-day-ahead variance and 1% VaR forecasts, all from data before the day forecast.

- HMM (two states, refitted each year-end): tomorrow's state chances are today's filtered chances
  times the transition matrix; the forecast distribution is a mixture of the states' normals.
- EWMA (lambda 0.94), and the 20-day standard deviation.
- GJR-GARCH(1,1) with Student-t shocks (arch), refitted each year-end and run with fixed parameters
  through the next year.
Returns are daily log returns in %.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.stats import t as student

from rh import hmm


def hmm_forecasts(x: pd.Series, K: int = 2, first: str = "1998-12-31") -> pd.DataFrame:
    """Per day from the year after `first`: forecast mean, variance and 1% VaR (as a return,
    negative) of that day's return from the HMM refitted at the previous year-end."""
    rt = hmm.realtime(x.to_numpy(), x.index, K, first=first)
    dates = rt["dates"]
    rows = []
    ends = sorted(rt["fits"])
    for i, d in enumerate(dates):
        e = max(k for k in ends if k < d)
        p = rt["fits"][e]
        prev = rt["probs"][i - 1] if i and max(k for k in ends if k < dates[i - 1]) == e \
            else None
        if prev is None:  # first day after a refit: filter up to the day before with this fit
            n = int(np.searchsorted(x.index, d))
            alpha, *_ = hmm.forward(x.to_numpy()[:n], p["pi"], p["A"], p["mu"], p["sd"])
            prev = alpha[-1]
        w = prev @ p["A"]
        m = float(w @ p["mu"])
        v = float(w @ (p["sd"] ** 2 + p["mu"] ** 2) - m * m)

        def cdf(q, w=w, p=p):
            return float(w @ norm.cdf((q - p["mu"]) / p["sd"])) - 0.01

        var = brentq(cdf, -100, 100)
        rows.append((d, m, v, var))
    return pd.DataFrame(rows, columns=["date", "mean", "var", "VaR"]).set_index("date")


def ewma(x: pd.Series, lam: float = 0.94) -> pd.Series:
    """EWMA variance forecast for each day from returns before it."""
    v = np.empty(len(x))
    v[0] = x.iloc[:20].var()
    r = x.to_numpy()
    for t in range(1, len(x)):
        v[t] = lam * v[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(v, index=x.index)


def gjr(x: pd.Series, first: str = "1998-12-31") -> pd.DataFrame:
    """GJR-GARCH(1,1)-t variance forecasts and 1% VaR, refitted at each year-end on the days up to
    it and run with those parameters through the next year."""
    from arch import arch_model

    dates = x.index
    ends = [d for d in pd.Series(dates, index=dates).groupby(dates.year).max()
            if d >= pd.Timestamp(first)]
    out = []
    for i, e in enumerate(ends):
        n = int(np.searchsorted(dates, e, side="right"))
        if n >= len(x):
            break
        stop = int(np.searchsorted(dates, ends[i + 1], side="right")) if i + 1 < len(ends) \
            else len(x)
        res = arch_model(x.iloc[:n], mean="Constant", p=1, o=1, q=1, dist="t").fit(disp="off")
        run = arch_model(x.iloc[:stop], mean="Constant", p=1, o=1, q=1, dist="t").fix(res.params)
        v = run.conditional_volatility.iloc[n:stop] ** 2
        nu = res.params["nu"]
        q = student.ppf(0.01, nu) * np.sqrt((nu - 2) / nu)
        out.append(pd.DataFrame({"var": v, "VaR": res.params["mu"] + q * np.sqrt(v)}))
    return pd.concat(out)


def qlike(r2: np.ndarray, h: np.ndarray) -> np.ndarray:
    """QLIKE loss of variance forecast h against the squared return (lower is better)."""
    return np.log(h) + r2 / h


def kupiec(breaches: np.ndarray, p: float = 0.01) -> float:
    """Kupiec likelihood-ratio statistic for the breach rate (chi-squared, 1 degree of freedom)."""
    n, k = len(breaches), int(breaches.sum())
    pi = k / n
    if k in (0, n):
        return float("nan")
    ll0 = (n - k) * np.log(1 - p) + k * np.log(p)
    ll1 = (n - k) * np.log(1 - pi) + k * np.log(pi)
    return float(-2 * (ll0 - ll1))
