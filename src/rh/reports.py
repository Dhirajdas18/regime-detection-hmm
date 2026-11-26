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


def nw_se_mean(e: np.ndarray, lags: int = 20) -> float:
    """Newey-West standard error of the mean of a daily series."""
    e = e - e.mean()
    n = len(e)
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k] / n)
    return float(np.sqrt(s / n))


def report_states_count(out_dir: Path) -> dict:
    """How many states? In-sample BIC and AIC for 1-5 states on 1993-2025; out of sample, each
    day's predictive log density from 1999 with the model refitted at each year-end on past days
    only (1-4 states; 1 state = constant volatility), the gain from each added state with a
    Newey-West t (20 lags) and the share of years it helped; and how far each state's volatility
    moved across the yearly refits, including four-state fits with a near-flat state."""
    from rh import hmm

    x = market()
    X = x.to_numpy()
    res = {}
    for K in range(1, 6):
        p = hmm.fit(X, K)
        res[f"in-sample, {K} state{'s' if K > 1 else ''}"] = (
            f"log-likelihood {p['ll']:.0f}, BIC {hmm.bic(p['ll'], K, len(X)):.0f}, AIC "
            f"{2 * hmm.n_params(K) - 2 * p['ll']:.0f}")
    rts = {K: hmm.realtime(X, x.index, K) for K in range(1, 5)}
    dates = rts[1]["dates"]
    res["out of sample"] = f"{len(dates)} days, {dates[0]:%b %Y}-{dates[-1]:%b %Y}"
    gains = {}
    for K in range(1, 5):
        res[f"  {K} state{'s' if K > 1 else ''}"] = (
            f"mean daily log score {rts[K]['logscore'].mean():.4f}")
    for a, b in ((1, 2), (2, 3), (3, 4)):
        diff = rts[b]["logscore"] - rts[a]["logscore"]
        t = diff.mean() / nw_se_mean(diff)
        yr = pd.Series(diff, index=dates).groupby(dates.year).sum()
        gains[b] = (diff.mean(), t, (yr > 0).mean())
        res[f"  {b} states against {a}"] = (f"+{diff.mean():.4f} a day (t {t:.2f}); better in "
                                             f"{(yr > 0).mean():.0%} of {len(yr)} years")
    vols = {}
    for K in (2, 3, 4):
        v = np.array([f["sd"] for f in rts[K]["fits"].values()]) * np.sqrt(PER_YEAR)
        vols[K] = v
        res[f"{K} states, volatility range across {len(v)} refits"] = ", ".join(
            f"state {k + 1} {v[:, k].min():.0f}-{v[:, k].max():.0f}%" for k in range(K))
    flat = [(e, f) for e, f in rts[4]["fits"].items() if f["sd"][0] * np.sqrt(PER_YEAR) < 5]
    res["  four-state refits with a near-flat state (volatility under 5% a year)"] = (
        f"{len(flat)} of {len(rts[4]['fits'])}; its volatility "
        f"{min(f['sd'][0] for _, f in flat) * np.sqrt(PER_YEAR):.1f}-"
        f"{max(f['sd'][0] for _, f in flat) * np.sqrt(PER_YEAR):.1f}% a year, chance of staying "
        f"{min(f['A'][0, 0] for _, f in flat):.2f}-{max(f['A'][0, 0] for _, f in flat):.2f}"
        if flat else "none")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ks = [2, 3, 4]
    v = [gains[k][0] for k in ks]
    bars = ax.bar([f"{k - 1} to {k}" for k in ks], v, color=plots.SERIES[0], width=0.55)
    for b_, k in zip(bars, ks, strict=True):
        ax.annotate(f"t {gains[k][1]:.1f}\nbetter in {gains[k][2]:.0%} of years",
                    (b_.get_x() + b_.get_width() / 2, b_.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=plots.TEXT)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_minor_formatter(lambda v, _: "")
    ax.set_ylabel("gain in daily predictive log score,\nout of sample 1999-2025")
    ax.set_xlabel("states")
    ax.margins(y=0.3)
    ax.set_title("Each extra state helped less", fontsize=10.5, loc="left")
    ax = axes[1]
    ends = list(rts[4]["fits"])
    for k, col in zip(range(3), (plots.SERIES[0], plots.SERIES[3], plots.SERIES[1]), strict=True):
        ax.plot(ends, vols[3][:, k], color=col, lw=2, label=f"3 states: state {k + 1}")
    ax.plot(ends, vols[4][:, 0], color=plots.TEXT_2, lw=1.4, ls="--",
            label="4 states: calmest state")
    ax.set_yscale("log")
    ax.set_yticks([0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.3%", "1%", "3%", "10%", "30%"])
    ax.set_ylabel("state's volatility a year, as refitted")
    ax.set_xlabel("refit at year-end")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("A fourth state was often a one-day quirk", fontsize=10.5, loc="left")
    fig.suptitle("Three states beat two out of sample in most years; a fourth added little",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "count")
    plt.close(fig)
    return res


def report_forecast(out_dir: Path) -> dict:
    """Does the regime improve risk forecasts? One-day-ahead variance and 1% VaR from data before
    each day (rh.forecast): the two-state HMM refitted each year-end, EWMA (0.94), the 20-day
    standard deviation and GJR-GARCH(1,1)-t refitted each year-end. (A) IIMA's market, 1999-2025.
    (B) NIFTY from the NSE bhavcopy archive, 2015-2026, adding India VIX at the previous close as a
    variance forecast (VIX² / 250). Scored by QLIKE (Newey-West t, 20 lags, of the difference
    from GJR) and 1% VaR breach rates with the Kupiec test (5% critical value 3.84); EWMA, 20-day
    and VIX VaR assume normal returns, the HMM uses its mixture, GJR its t."""
    from rh import forecast as F
    from rh import lake

    def score(r, fc, var):
        out, base = {}, F.qlike(r.to_numpy() ** 2 + 1e-12, fc["GJR-t"].to_numpy())
        for k, v in fc.items():
            L = F.qlike(r.to_numpy() ** 2 + 1e-12, v.to_numpy())
            d = L - base
            t = d.mean() / nw_se_mean(d) if k != "GJR-t" else float("nan")
            b = (r < var[k]).to_numpy()
            out[k] = (L.mean(), d.mean(), t, b.mean(), F.kupiec(b))
        return out

    x = market()
    h, g, e = F.hmm_forecasts(x), F.gjr(x), F.ewma(x)
    s20 = x.rolling(20).var().shift(1)
    idx = h.index.intersection(g.index)
    r = x.loc[idx]
    A = score(r, {"HMM": h.loc[idx, "var"], "GJR-t": g.loc[idx, "var"], "EWMA": e.loc[idx],
                  "20-day": s20.loc[idx]},
              {"HMM": h.loc[idx, "VaR"], "GJR-t": g.loc[idx, "VaR"],
               "EWMA": -2.3263 * np.sqrt(e.loc[idx]), "20-day": -2.3263 * np.sqrt(s20.loc[idx])})
    res = {"A: IIMA market": f"{len(idx)} days, {idx[0]:%b %Y}-{idx[-1]:%b %Y}"}
    for k, (ql, d, t, br, ku) in A.items():
        res[f"  {k}"] = (f"QLIKE {ql:.4f}" + (f" ({d:+.4f} against GJR-t, t {t:.2f})"
                                              if k != "GJR-t" else "")
                         + f"; 1% VaR breached on {br:.2%} of days (Kupiec {ku:.1f})")
    m = lake.nifty_vix()
    xn = m["r"]
    hn, gn, en = F.hmm_forecasts(xn, first="2014-12-31"), F.gjr(xn, first="2014-12-31"), F.ewma(xn)
    vixv = (m["vix"].shift(1) ** 2 / PER_YEAR).dropna()
    idn = hn.index.intersection(gn.index).intersection(vixv.index)
    rn = xn.loc[idn]
    B = score(rn, {"HMM": hn.loc[idn, "var"], "GJR-t": gn.loc[idn, "var"], "EWMA": en.loc[idn],
                   "India VIX": vixv.loc[idn]},
              {"HMM": hn.loc[idn, "VaR"], "GJR-t": gn.loc[idn, "VaR"],
               "EWMA": -2.3263 * np.sqrt(en.loc[idn]),
               "India VIX": -2.3263 * np.sqrt(vixv.loc[idn])})
    res["B: NIFTY"] = (f"{len(idn)} days with India VIX, {idn[0]:%b %Y}-{idn[-1]:%b %Y}; HMM "
                       "and GJR first fitted on 2013-2014")
    for k, (ql, d, t, br, ku) in B.items():
        res[f"  {k} "] = (f"QLIKE {ql:.4f}" + (f" ({d:+.4f} against GJR-t, t {t:.2f})"
                                               if k != "GJR-t" else "")
                          + f"; 1% VaR breached on {br:.2%} of days (Kupiec {ku:.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, data_, title in ((axes[0], (A, B), "QLIKE against GJR-t (above 0 = worse)"),
                             (axes[1], (A, B), "1% VaR: share of days breached")):
        labels = ["HMM", "EWMA", "20-day", "India VIX", "GJR-t"]
        x_ = np.arange(len(labels))
        for i, (dd, nm, col) in enumerate(zip(data_, ("IIMA market 1999-2025", "NIFTY 2015-2026"),
                                              (plots.MUTED, plots.SERIES[0]), strict=True)):
            pos = [j for j, k in enumerate(labels) if k in dd]
            vals = [dd[labels[j]][1] if ax is axes[0] else 100 * dd[labels[j]][3] for j in pos]
            ax.bar(x_[pos] + (i - 0.5) * 0.38, vals, 0.36, color=col, label=nm)
        ax.set_xticks(x_, labels, fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0].axhline(0, color=plots.TEXT_2, lw=0.8)
    axes[0].set_ylabel("QLIKE difference from GJR-t")
    axes[1].axhline(1, color=plots.TEXT_2, lw=1, ls="--")
    axes[1].text(4.45, 1.03, "1% target", fontsize=8.5, color=plots.TEXT_2, ha="right")
    axes[1].set_ylabel("% of days")
    fig.suptitle("The regime model forecast volatility worse than GJR-GARCH and India VIX",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "forecast")
    plt.close(fig)
    return res


def report_allocate(out_dir: Path) -> dict:
    """Stocks or cash by regime, after costs (rh.allocate), IIMA's market 1999-2025, every rule
    deciding at a close from data up to it: buy and hold; HMM switch (cash when the real-time
    turbulent chance is above 50%); HMM with a band (cash above 70%, back below 30%, fixed in
    advance); HMM scaled (weight = calm chance, traded when it moves over 0.1); volatility target
    (GJR forecast refitted yearly, target = the 1993-1998 volatility, no leverage, traded when the
    weight moves over 0.1); and the 200-day trend (in the market when above its 200-day average).
    Sharpe ratios above cash, net and gross of costs, by half."""
    from rh import allocate as Al
    from rh import data, hmm
    from rh import forecast as F

    x = market()
    m = data.factors("daily")
    rt = hmm.realtime(x.to_numpy(), x.index, 2)
    idx = rt["dates"]
    pt = pd.Series(rt["probs"][:, 1], index=idx)
    mkt, rf = m["MARKET"].loc[idx], m["RF"].loc[idx]
    target = x.loc[:"1998"].std()
    volw = (target / np.sqrt(F.gjr(x)["var"].reindex(idx))).clip(upper=1)
    lvl = np.exp(x.cumsum() / 100)
    trend = (lvl > lvl.rolling(200).mean()).astype(float).reindex(idx)
    rules = {"buy and hold": pd.Series(1.0, index=idx),
             "HMM switch (50%)": (pt <= 0.5).astype(float),
             "HMM band (70/30)": Al.band(pt),
             "HMM scaled": Al.sticky(1 - pt),
             "volatility target (GJR)": Al.sticky(volw),
             "200-day trend": trend}
    res = {"scored": f"{len(idx)} days, {idx[0]:%b %Y}-{idx[-1]:%b %Y}; volatility target "
                     f"{target * np.sqrt(PER_YEAR):.1f}% a year (1993-1998)"}
    out, curves = {}, {}
    for k, w in rules.items():
        r = Al.run(w, mkt, rf)
        s_, sg = Al.stats(r["net"], rf, r["traded"]), Al.stats(r["gross"], rf)
        h1 = Al.stats(r["net"].loc[:"2012"], rf.loc[:"2012"])
        h2 = Al.stats(r["net"].loc["2013":], rf.loc["2013":])
        out[k] = (s_["sharpe"], sg["sharpe"])
        curves[k] = np.exp(np.log1p(r["net"]).cumsum())
        res[k] = (f"compounded {s_['cagr']:+.1%} a year after costs, volatility {s_['vol']:.1%}, "
                  f"Sharpe {s_['sharpe']:.2f} ({sg['sharpe']:.2f} before costs), worst fall "
                  f"{s_['dd']:.0%}; traded {s_['turnover']:.1f}x a year, costs "
                  f"{r['cost'].sum() / (len(r) / PER_YEAR):.2%} a year, in the market "
                  f"{r['weight'].mean():.0%}; Sharpe 1999-2012 {h1['sharpe']:.2f}, 2013-2025 "
                  f"{h2['sharpe']:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    show = {"buy and hold": plots.TEXT, "HMM band (70/30)": plots.SERIES[1],
            "volatility target (GJR)": plots.SERIES[0], "200-day trend": plots.SERIES[2]}
    for k, col in show.items():
        ax.plot(curves[k].index, curves[k], color=col, lw=1.6 if k != "buy and hold" else 1.2,
                label=k)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda v, _: f"₹{v:g}")
    ax.set_ylabel("₹1 from 1999, after costs")
    ax.set_title("Growth after costs", fontsize=10.5, loc="left")
    ax = axes[1]
    keys = list(rules)
    y = np.arange(len(keys))[::-1]
    ax.barh(y + 0.18, [out[k][1] for k in keys], 0.34, color=plots.GRID, label="before costs")
    ax.barh(y - 0.18, [out[k][0] for k in keys], 0.34, color=plots.SERIES[0], label="after costs")
    for yy, k in zip(y, keys, strict=True):
        ax.text(out[k][0] + 0.01, yy - 0.18, f"{out[k][0]:.2f}", va="center", fontsize=8.5,
                color=plots.TEXT)
    ax.axvline(out["buy and hold"][0], color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_yticks(y, keys, fontsize=9)
    ax.set_xlabel("Sharpe ratio above cash, 1999-2025")
    ax.set_xlim(0, 0.95)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Costs took the HMM rules below buy and hold", fontsize=10.5, loc="left")
    fig.suptitle("Switching on the regime lost to buy and hold after costs; simpler rules did "
                 "better",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "allocate")
    plt.close(fig)
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "states": report_states,
    "lookahead": report_lookahead,
    "count": report_states_count,
    "forecast": report_forecast,
    "allocate": report_allocate,
}
