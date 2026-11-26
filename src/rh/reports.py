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


def toy_switch(turbulent: pd.Series, mkt: pd.Series, rf: pd.Series) -> pd.Series:
    """Daily return of holding the market on days after a calm close and cash (the risk-free rate)
    after a turbulent one; no costs."""
    invest = (~turbulent).shift(1, fill_value=True)
    return pd.Series(np.where(invest, mkt, rf), index=mkt.index)


def sharpe(r: pd.Series, rf: pd.Series) -> float:
    e = r - rf
    return float(e.mean() / e.std() * np.sqrt(PER_YEAR))


def report_lookahead(out_dir: Path) -> dict:
    """Three ways to call the state, from 1999 (two states): hindsight (smoothed and Viterbi, model
    fitted on 1993-2025); filtered with that whole-period model; and truly real time (refitted
    each year-end on the days up to it, then filtered through the next year). How often each
    agrees with hindsight, how late real time flags turbulent spells of 20+ days, false alarms, and
    a toy rule - market after a calm close, cash after a turbulent one, no costs - with Sharpe
    ratios above the risk-free rate."""
    from rh import data, hmm

    x = market()
    X = x.to_numpy()
    p = hmm.fit(X, 2)
    sm, _, _ = hmm.smooth(X, p["pi"], p["A"], p["mu"], p["sd"])
    fi, *_ = hmm.forward(X, p["pi"], p["A"], p["mu"], p["sd"])
    vi = hmm.viterbi(X, p["pi"], p["A"], p["mu"], p["sd"])
    rt = hmm.realtime(X, x.index, 2)
    idx = rt["dates"]
    full = pd.DataFrame({"smoothed": sm[:, 1] > 0.5, "Viterbi": vi == 1,
                         "filtered": fi[:, 1] > 0.5}, index=x.index).loc[idx]
    full["real time"] = rt["probs"][:, 1] > 0.5
    res = {"scored": f"{len(idx)} days, {idx[0]:%d %b %Y}-{idx[-1]:%d %b %Y}; real-time model "
                     f"refitted {len(rt['fits'])} times"}
    for k in ("Viterbi", "filtered", "real time"):
        res[f"  {k}: same state as smoothed"] = (f"{(full[k] == full['smoothed']).mean():.1%} of "
                                                  f"days; turbulent {full[k].mean():.0%} of days")
    ep = runs(full["smoothed"])
    ep = ep[ep["days"] >= 20]
    for k in ("filtered", "real time"):
        lags = []
        for r in ep.itertuples():
            hit = np.flatnonzero(full[k].loc[r.start:r.end].to_numpy())
            lags.append(hit[0] if len(hit) else np.nan)
        lags = np.array(lags, float)
        rr = runs(full[k])
        fa = sum(1 for r in rr.itertuples() if not full["smoothed"].loc[r.start:r.end].any())
        res[f"{k}: hindsight turbulent spells of 20+ days caught"] = (
            f"{np.isfinite(lags).sum()} of {len(ep)}, median {np.nanmedian(lags):.0f} trading days "
            f"late (mean {np.nanmean(lags):.1f}); {fa} of its {len(rr)} turbulent runs touched no "
            "hindsight turbulent day")
    m = data.factors("daily").loc[idx]
    mkt, rf = m["MARKET"], m["RF"]
    toy = {k: toy_switch(full[k], mkt, rf) for k in full}
    toy["buy and hold"] = mkt
    sh = {}
    for k, r in toy.items():
        sh[k] = sharpe(r, rf)
        g = float(np.exp(np.log1p(r).sum() * PER_YEAR / len(r)) - 1)
        res[f"toy rule, {k}"] = (f"compounded {g:+.1%} a year, volatility "
                                 f"{r.std() * np.sqrt(PER_YEAR):.1%}, Sharpe {sh[k]:.2f}")
    fits = rt["fits"]
    res["real-time model's turbulent volatility over the years"] = ", ".join(
        f"{e:%Y} {f['sd'][1] * np.sqrt(PER_YEAR):.0f}%" for e, f in list(fits.items())[::5])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    win = slice("2007-10-01", "2009-12-31")
    smp = pd.Series(sm[:, 1], index=x.index).loc[win]
    rtp = pd.Series(rt["probs"][:, 1], index=idx).loc[win]
    ax.fill_between(smp.index, smp, color=plots.SERIES[1], alpha=0.25, lw=0,
                    label="hindsight (smoothed)")
    ax.plot(rtp.index, rtp, color=plots.SERIES[0], lw=1.2, label="real time (refitted yearly)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_ylabel("chance of the turbulent state")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 0.99), ncols=2)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("2008-2009: hindsight is sure, real time flickers", fontsize=10.5, loc="left",
                 pad=22)
    ax = axes[1]
    keys = list(toy)
    cols = [plots.SERIES[1], plots.SERIES[1], plots.MUTED, plots.SERIES[0], plots.TEXT_2]
    y = np.arange(len(keys))[::-1]
    ax.barh(y, [sh[k] for k in keys], color=cols, height=0.6)
    for yy, k in zip(y, keys, strict=True):
        ax.text(sh[k] + 0.03, yy, f"{sh[k]:.2f}", va="center", fontsize=9, color=plots.TEXT)
    ax.set_yticks(y, ["hindsight (smoothed)", "hindsight (Viterbi)", "filtered, whole-period model",
                      "real time", "buy and hold"])
    ax.set_xlabel("Sharpe ratio above cash, 1999-2025 (no costs)")
    ax.set_xlim(0, max(sh.values()) * 1.25)
    ax.set_title("A cash-when-turbulent rule", fontsize=10.5, loc="left")
    fig.suptitle("Hindsight labels made regime switching look three times better than it was",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "lookahead")
    plt.close(fig)
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "states": report_states,
    "lookahead": report_lookahead,
}
