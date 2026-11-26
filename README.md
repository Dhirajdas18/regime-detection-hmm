# Regime Detection HMM

[![CI](https://github.com/Dhirajdas18/regime-detection-hmm/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhirajdas18/regime-detection-hmm/actions/workflows/ci.yml)
[![Regime lab](https://github.com/Dhirajdas18/regime-detection-hmm/actions/workflows/pages.yml/badge.svg)](https://dhirajdas18.github.io/regime-detection-hmm/)

Hidden Markov models of calm and turbulent regimes in the Indian stock market, on IIM Ahmedabad's
daily market returns from 1993 to 2025 and, for comparisons with what an investor can trade, NIFTY,
India VIX and stocks from a local NSE archive (from 2013; only derived results committed): what the
states look like, the look-ahead trap in hindsight labels, how many states, volatility forecasts,
regime-switching allocation after costs, and factors by regime.

**Live regime lab: [dhirajdas18.github.io/regime-detection-hmm](https://dhirajdas18.github.io/regime-detection-hmm/)**,
pick any years and two or three states; the page fits the model in your browser and shows when
the market was calm or turbulent, as seen at the time and with hindsight.

## Features

- [x] Two and three states on daily market returns: their returns, volatility and persistence
- [x] The look-ahead trap: real-time (filtered) against hindsight (smoothed, Viterbi) states
- [x] How many states, and do they keep their meaning across refits, out of sample
- [ ] Volatility and VaR forecasts from the regime against EWMA, GJR-GARCH and India VIX
- [ ] Regime-switching between stocks and cash after costs, against volatility targeting and trend
- [ ] Size, value and momentum by regime, out of sample
- [ ] Verdict: does an HMM regime earn a place next to the reference model's current risk inputs

## Findings

### 1. Indian stocks switched between a calm and a turbulent state that lasted weeks

<a href="docs/img/states.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/states.png"><img src="docs/img/states.gif" alt="Growth of one rupee in the Indian market from 1993 to 2025 with the turbulent state shaded, and the return distributions of three fitted states" width="720"></picture></a>

IIMA's daily market return (the market premium plus the risk-free rate, survivorship-bias
adjusted) as a log return, 8,002 trading days from October 1993 to December 2025. A Gaussian hidden
Markov model gives each day one of two or three hidden states, each with its own average and
volatility and a fixed chance of switching each day; it is fitted by expectation-maximisation from
four starts, and states are sorted from calmest to most turbulent. Everything here is fitted on
the whole period and the shaded episodes use smoothed probabilities, which see the future; finding
2 deals with that. For comparison, a simple rule marks days whose 20-day volatility is above the
level that gives the same share of turbulent days. Annual figures use 250 trading days.

- **Two states: calm and turbulent.** The calm state had 12.6% volatility a year and lasted 44 days
  on average, 70% of days; the turbulent state 34.1% volatility, 19 days, 30% of days. Their average
  returns were +30% and −30% a year, which describes the states with hindsight, not a forecast.
- **Three states split off a crash state.** Volatility of 11.0%, 21.5% and 48.6% a year, lasting 38,
  21 and 14 days, on 50%, 41% and 9% of days; the crash state averaged −89% a year.
- **The long turbulent spells were the big crashes.** The longest were May 2008 to August 2009 (309
  trading days), December 1999 to June 2000 (128) and February to August 1998 (112).
- **A simple volatility rule saw much the same.** With a threshold of 21.0% a year it agreed with the
  two-state model on 84% of days and gave 98 turbulent runs (median 16 days) against the model's 100
  (median 12).

### 2. Hindsight labels made regime switching look three times better than it was

<a href="docs/img/lookahead.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/lookahead.png"><img src="docs/img/lookahead.gif" alt="Hindsight and real-time chance of the turbulent state through 2008-2009, and the Sharpe ratio of a cash-when-turbulent rule under hindsight, filtered and real-time states against buy and hold, 1999-2025" width="720"></picture></a>

The two-state model of finding 1, scored from January 1999 to December 2025 (6,720 days) in three
ways. Hindsight: smoothed probabilities and the Viterbi path, from a model fitted on 1993–2025, so
both the labels and the model see the future. Filtered: probabilities from only the days up to each
date, but from that same whole-period model. Real time: the model refitted at each year-end (27
times) on the days up to it, then filtered through the next year, so nothing after a day is used
on that day. A state counts as turbulent above 50%. A toy rule holds the market on days after a calm
close and cash (IIMA's risk-free rate) after a turbulent one, with no trading costs (finding 5 adds
them); Sharpe ratios are measured above the risk-free rate.

- **Hindsight looked excellent.** The toy rule's Sharpe ratio was 1.71 with smoothed states and
  1.51 with the Viterbi path, compounding 27.6% and 25.4% a year.
- **Real time was ordinary.** With real-time states it was 0.56 (14.4% a year, volatility 14.8%),
  with filtered states from the whole-period model 0.55, and buy and hold 0.51 (16.1% a year,
  volatility 21.4%): lower risk for lower return.
- **Real time was a few days late.** It caught 29 of the 30 hindsight turbulent spells of 20 days or
  more, a median of 3 trading days after they began (mean 4.4); 23 of its 219 turbulent runs never
  touched a hindsight turbulent day.
- **Real time and hindsight agreed on most days, just not the ones that matter.** Real-time states
  matched smoothed ones on 88.5% of days and flagged 19% of days as turbulent against 28%; in
  2008–2009 the real-time chance flickered between near 0 and near 100% while hindsight was sure.
- **The model itself barely changed.** Refitted every year, its turbulent state's volatility ranged
  from 32% to 41% a year.

### 3. Three states beat two out of sample in most years; a fourth added little

<a href="docs/img/count.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/count.png"><img src="docs/img/count.gif" alt="Out-of-sample gain in daily predictive log score from each added state, and each state's volatility across yearly refits for three states and the calmest of four" width="720"></picture></a>

In sample, the information criteria BIC and AIC for 1 to 5 states fitted on 1993–2025. Out of
sample, finding 2's real-time scheme for 1 to 4 states: the model refitted at each year-end on
past days only, then each day from January 1999 to December 2025 (6,720 days) scored by the
log of the density the model gave that day's return before seeing it (higher is better; one state
is a constant-volatility normal). Each added state's daily gain gets a Newey-West t (20 lags) and
the share of the 27 years in which it helped. The refits also show how far each state's volatility
moved from year to year.

- **More states always fitted better in sample.** BIC fell from 27,644 (one state) to 24,928, 24,546,
  24,431 and 24,352 (five), so it never chose a number.
- **Two states were a big step out of sample, three a real one, four marginal.** Two beat one by
  0.194 a day (t 13.25) in all 27 years; three beat two by 0.025 (t 7.85) in 93% of years; four beat
  three by 0.008 (t 3.14) in only 52% of years.
- **The fourth state was often not a regime.** In 11 of 27 refits the four-state model's calmest
  state had a volatility of 0.3–3.0% a year and a chance of staying of 0.08–0.59 a day: it was
  absorbing the many very small daily moves, a quirk of the return distribution, not a market mood.
- **Two states were the most stable.** Across refits the calm state's volatility ran 13–17% and the
  turbulent 32–41%; with three states the middle state ran 12–27%, close to the calm one before
  2004.

## Quick start

```bash
uv sync
uv run python scripts/fetch_data.py   # IIMA factor library -> data/iima/
uv run rh report states               # topics: states, lookahead, count
uv run python scripts/build_site.py   # regime lab -> site/index.html
uv run pytest
```

The data files are downloaded, not committed.

## Design

- `src/rh/data.py`: finds the IIMA library's latest release and downloads its survivorship-bias
  adjusted daily and monthly files.
- `src/rh/hmm.py`: the Gaussian hidden Markov model: scaled forward and backward passes, filtered
  and smoothed probabilities, Viterbi, Baum-Welch from several starts with states sorted by
  volatility, BIC, and real-time probabilities refitted each year-end; checked against hmmlearn in
  the tests.
- `src/rh/rh.js`: the lab's browser copy of the fit and the filter, tested against Python through
  Node.
- `src/rh/site.py`: the regime lab, one static Plotly page carrying the daily market returns.
- `src/rh/reports.py`: README findings, reproducible with `rh report <topic>`.
- Conventions: daily log returns in percent; annual means are 250 times the daily mean, annual
  volatility √250 times the daily.
- Data: IIM Ahmedabad's Indian Fama-French-momentum data library (Agarwalla, Jacob and Varma, 2013,
  Four factor model in Indian equities market, IIMA Working Paper 2013-09-05), daily market return.
- Charts: one validated palette; README charts are animated GIFs with a static fallback for
  reduced-motion users.

Research only, not investment advice. MIT licence.
