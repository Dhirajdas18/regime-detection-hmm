"""A Gaussian hidden Markov model for one return series, mirrored in rh.js.

Each day the market is in one of K hidden states; state k's return is normal with mean mu_k and
standard deviation sigma_k; tomorrow's state depends only on today's through the transition matrix
A (A[i, j] = chance of moving from i to j). Fitted by expectation-maximisation (Baum-Welch) from a
few starting points, keeping the best likelihood. States are always sorted by volatility, calmest
first, so a label means the same thing in every fit.

Three ways to read the states, which must not be confused:
- filtered: P(state on day t | returns up to day t) - what was knowable that day;
- smoothed: P(state on day t | all returns) - uses the future;
- Viterbi: the single most likely path given all returns - uses the future.
"""

import numpy as np

LOG2PI = np.log(2 * np.pi)


def _dens(x, mu, sd):
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    return np.exp(-0.5 * z * z) / (sd[None, :] * np.sqrt(2 * np.pi))


def forward(x, pi, A, mu, sd):
    """Scaled forward pass: filtered probabilities (T x K), scale factors, log-likelihood."""
    B = _dens(x, mu, sd)
    T, K = B.shape
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi * B[0]
    for t in range(T):
        if t:
            a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c, float(np.log(c).sum()), B


def smooth(x, pi, A, mu, sd):
    """Smoothed probabilities (T x K), pairwise sums (K x K) and log-likelihood."""
    alpha, c, ll, B = forward(x, pi, A, mu, sd)
    T, K = B.shape
    beta = np.ones((T, K))
    xi = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        bb = B[t + 1] * beta[t + 1]
        beta[t] = (A @ bb) / c[t + 1]
        xi += A * np.outer(alpha[t], bb) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, xi, ll


def viterbi(x, pi, A, mu, sd):
    """Most likely state path given all returns."""
    lB = np.log(_dens(x, mu, sd) + 1e-300)
    lA = np.log(A + 1e-300)
    T, K = lB.shape
    d = np.log(pi + 1e-300) + lB[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        m = d[:, None] + lA
        back[t] = m.argmax(axis=0)
        d = m.max(axis=0) + lB[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(d.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _order(p):
    o = np.argsort(p["sd"])
    return {"pi": p["pi"][o], "A": p["A"][np.ix_(o, o)], "mu": p["mu"][o], "sd": p["sd"][o],
            "ll": p["ll"], "iters": p["iters"]}


def _start(x, K, rng, jitter):
    """Start from volatility quantiles: split days by |x| into K groups."""
    q = np.quantile(np.abs(x), np.linspace(0, 1, K + 1))
    sd = np.array([max(np.std(x[(np.abs(x) >= q[k]) & (np.abs(x) <= q[k + 1])]), 1e-6)
                   for k in range(K)])
    sd = sd * np.exp(jitter * rng.normal(size=K))
    A = np.full((K, K), 0.05 / max(K - 1, 1))
    np.fill_diagonal(A, 0.95)
    return {"pi": np.full(K, 1 / K), "A": A, "mu": np.full(K, x.mean()), "sd": sd}


def fit(x, K: int = 2, restarts: int = 4, iters: int = 1000, tol: float = 1e-6,
        seed: int = 0, sd_floor: float = 1e-4) -> dict:
    """Baum-Welch from `restarts` starts (the first without jitter); keeps the highest
    likelihood. Returns pi, A, mu, sd (sorted calmest first), ll and iterations."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        p = _start(x, K, rng, 0.0 if r == 0 else 0.3)
        prev = -np.inf
        for it in range(iters):  # noqa: B007 (it is used after the loop)
            gamma, xi, ll = smooth(x, p["pi"], p["A"], p["mu"], p["sd"])
            w = gamma.sum(axis=0)
            p["pi"] = gamma[0]
            p["A"] = xi / xi.sum(axis=1, keepdims=True)
            p["mu"] = gamma.T @ x / w
            p["sd"] = np.maximum(np.sqrt((gamma * (x[:, None] - p["mu"]) ** 2).sum(axis=0) / w),
                                 sd_floor)
            if ll - prev < tol:  # absolute: stop once the log-likelihood gains < tol
                break
            prev = ll
        p["ll"], p["iters"] = ll, it + 1
        if best is None or p["ll"] > best["ll"]:
            best = p
    return _order(best)


def n_params(K: int) -> int:
    return K * (K - 1) + 2 * K + (K - 1)


def bic(ll: float, K: int, T: int) -> float:
    return -2 * ll + n_params(K) * np.log(T)


def stationary(A):
    vals, vecs = np.linalg.eig(A.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def realtime(x, dates, K: int = 2, first: str = "1998-12-31", restarts: int = 2) -> dict:
    """Truly real-time probabilities: at each year-end from `first`, refit on all days up to it,
    then filter through the next year with those parameters (the filter itself starts from the
    first day each time, so its state on day 1 of the year uses all earlier days). Returns the
    filtered probabilities (rows from the day after `first`), each day's predictive log density
    (log of the forward pass's scale factor: the density of that day's return given all earlier
    days, under the model fitted before that year) and each year's fit."""
    import pandas as pd

    x = np.asarray(x, float)
    dates = pd.DatetimeIndex(dates)
    ends = [d for d in pd.Series(dates, index=dates).groupby(dates.year).max()
            if d >= pd.Timestamp(first)]
    probs, logc, fits = [], [], {}
    for i, e in enumerate(ends):
        n = int(np.searchsorted(dates, e, side="right"))
        if n >= len(x):
            break
        p = fit(x[:n], K, restarts=restarts)
        fits[e] = p
        stop = int(np.searchsorted(dates, ends[i + 1], side="right")) if i + 1 < len(ends) \
            else len(x)
        alpha, c, *_ = forward(x[:stop], p["pi"], p["A"], p["mu"], p["sd"])
        probs.append(alpha[n:stop])
        logc.append(np.log(c[n:stop]))
    start = int(np.searchsorted(dates, ends[0], side="right"))
    return {"probs": np.vstack(probs), "dates": dates[start:start + sum(len(q) for q in probs)],
            "fits": fits, "logscore": np.concatenate(logc)}
