"""README findings, one function per topic: `rh report <topic>` writes charts to out/."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rh import plots

PER_YEAR = 250


def market() -> pd.Series:
    """IIMA's daily market return (market premium + risk-free), as daily log returns in %."""
    from rh import data

    m = data.factors("daily")
    return (np.log1p(m["MARKET"]) * 100).rename("market")


def runs(flag: pd.Series) -> pd.DataFrame:
    """Consecutive runs of True in a boolean series: start, end, length in days."""
    f = flag.astype(int).to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, f, 0]))
    starts, ends = edges[::2], edges[1::2] - 1
    return pd.DataFrame({"start": flag.index[starts], "end": flag.index[ends],
                         "days": ends - starts + 1})


def report_states(out_dir: Path) -> dict:
    """Two- and three-state Gaussian HMMs on IIMA's daily market returns: each state's average
    and volatility a year, how long it lasts, how often it holds; the turbulent episodes (smoothed,
    so with hindsight); and a simple rule for comparison, 20-day volatility above the level that
    gives the same share of turbulent days."""
    from rh import hmm

    x = market()
    res = {"data": f"{len(x)} days, {x.index[0]:%d %b %Y}-{x.index[-1]:%d %b %Y}"}
    fits = {}
    for K in (2, 3):
        p = hmm.fit(x.to_numpy(), K)
        fits[K] = p
        share = hmm.stationary(p["A"])
        res[f"{K} states (log-likelihood {p['ll']:.0f})"] = "; ".join(
            f"state {k + 1}: mean {p['mu'][k] * PER_YEAR:+.0f}% a year, volatility "
            f"{p['sd'][k] * np.sqrt(PER_YEAR):.1f}%, stays {1 / (1 - p['A'][k, k]):.0f} days on "
            f"average, {share[k]:.0%} of days" for k in range(K))
    p = fits[2]
    g, _, _ = hmm.smooth(x.to_numpy(), p["pi"], p["A"], p["mu"], p["sd"])
    turb = pd.Series(g[:, 1] > 0.5, index=x.index)
    ep = runs(turb)
    long = ep[ep["days"] >= 60].sort_values("days", ascending=False)
    res["turbulent episodes of 60+ days (2 states, smoothed)"] = f"{len(long)}: " + "; ".join(
        f"{r.start:%b %Y}-{r.end:%b %Y} ({r.days} days)" for r in long.head(8).itertuples())
    vol20 = x.rolling(20).std()
    cut = vol20.quantile(1 - turb.mean())
    rule = (vol20 > cut).loc[vol20.notna()]
    rr = runs(rule)
    agree = (rule == turb.loc[rule.index]).mean()
    res["20-day volatility rule, same turbulent share"] = (
        f"threshold {cut * np.sqrt(PER_YEAR):.1f}% a year; agrees with the HMM on {agree:.0%} of "
        f"days; {len(rr)} turbulent runs (median {rr['days'].median():.0f} days) against the HMM's "
        f"{len(ep)} (median {ep['days'].median():.0f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    wealth = np.exp(x.cumsum() / 100)
    ax.plot(wealth.index, wealth, color=plots.TEXT, lw=1.2)
    ax.set_yscale("log")
    for r in ep.itertuples():
        ax.axvspan(r.start, r.end, color=plots.SERIES[1], alpha=0.25, lw=0)
    ax.set_ylabel("₹1 in the Indian market (log scale)")
    ax.yaxis.set_major_formatter(lambda v, _: f"₹{v:g}")
    ax.set_title("Turbulent state shaded (two states, with hindsight)", fontsize=10.5, loc="left")
    ax = axes[1]
    xs = np.linspace(-8, 8, 400)
    ax.hist(x.clip(-8, 8), bins=160, density=True, color=plots.GRID)
    p3 = fits[3]
    share3 = hmm.stationary(p3["A"])
    names = ("calm", "middle", "turbulent")
    for k, col in zip(range(3), (plots.SERIES[0], plots.SERIES[3], plots.SERIES[1]), strict=True):
        dens = share3[k] * np.exp(-0.5 * ((xs - p3["mu"][k]) / p3["sd"][k]) ** 2) / (
            p3["sd"][k] * np.sqrt(2 * np.pi))
        ax.plot(xs, dens, color=col, lw=2,
                label=f"{names[k]}: {p3['sd'][k] * np.sqrt(PER_YEAR):.0f}% vol, {share3[k]:.0%}")
    ax.set_xlabel("daily return, %")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Three states' return distributions", fontsize=10.5, loc="left")
    fig.suptitle("Indian stocks switched between a calm and a turbulent state that lasted weeks",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "states")
    plt.close(fig)
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "states": report_states,
}
