"""Stocks-or-cash rules on IIMA's daily market return, each deciding at a day's close from data up
to it and holding the weight over the next day; cash earns the risk-free rate.

Costs per rupee traded, as in momentum-after-costs (Zerodha delivery, Nov 2026): STT 0.1% each
side, stamp duty 0.015% on buys, NSE 0.00322% and SEBI 0.0001% each side with 18% GST, and 5 bp of
slippage each side. That is equity delivery; an index ETF pays less STT, so these are on the high
side. No tax.
"""

import numpy as np
import pandas as pd

BUY = 0.001 + 0.00015 + (0.0000322 + 0.000001) * 1.18 + 0.0005
SELL = 0.001 + (0.0000322 + 0.000001) * 1.18 + 0.0005


def run(weight: pd.Series, mkt: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """Daily net return of holding `weight` (decided at the previous close) in the market and the
    rest in cash, paying costs on each change of weight."""
    w = weight.reindex(mkt.index).ffill().fillna(1.0).clip(0, 1)
    held = w.shift(1, fill_value=w.iloc[0])
    dw = held.diff().fillna(0.0)
    cost = dw.clip(lower=0) * BUY + (-dw).clip(lower=0) * SELL
    gross = held * mkt + (1 - held) * rf
    return pd.DataFrame({"weight": held, "gross": gross, "net": gross - cost, "cost": cost,
                         "traded": dw.abs()})


def band(prob: pd.Series, hi: float = 0.7, lo: float = 0.3) -> pd.Series:
    """Weight 0 after the turbulent chance rises above `hi`, back to 1 after it falls below `lo`."""
    out, w = [], 1.0
    for p in prob.to_numpy():
        if w == 1.0 and p > hi:
            w = 0.0
        elif w == 0.0 and p < lo:
            w = 1.0
        out.append(w)
    return pd.Series(out, index=prob.index)


def sticky(target: pd.Series, step: float = 0.1) -> pd.Series:
    """Follow a target weight, trading only when it has moved more than `step` from the holding."""
    out, w = [], float(target.iloc[0])
    for t in target.to_numpy():
        if abs(t - w) > step:
            w = float(t)
        out.append(w)
    return pd.Series(out, index=target.index)


def stats(r: pd.Series, rf: pd.Series, traded: pd.Series | None = None,
          per_year: int = 250) -> dict:
    e = r - rf
    wealth = np.exp(np.log1p(r).cumsum())
    out = {"cagr": float(wealth.iloc[-1] ** (per_year / len(r)) - 1),
           "vol": float(r.std() * np.sqrt(per_year)),
           "sharpe": float(e.mean() / e.std() * np.sqrt(per_year)),
           "dd": float((wealth / wealth.cummax() - 1).min())}
    if traded is not None:
        out["turnover"] = float(traded.sum() / (len(r) / per_year))
    return out
