import numpy as np
import pandas as pd

from rh import allocate as Al


def test_run_holds_yesterdays_weight_and_pays_costs_on_changes():
    idx = pd.bdate_range("2020-01-01", periods=5)
    mkt = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0], index=idx)
    rf = pd.Series(0.0001, index=idx)
    w = pd.Series([1, 1, 0, 0, 1], index=idx, dtype=float)
    r = Al.run(w, mkt, rf)
    assert list(r["weight"]) == [1, 1, 1, 0, 0]  # decided at a close, held the next day
    assert np.isclose(r["cost"].iloc[3], Al.SELL)
    assert np.isclose(r["net"].iloc[3], 0.0001 - Al.SELL)
    assert np.isclose(r["net"].iloc[2], -0.01)


def test_band_needs_both_thresholds_and_sticky_ignores_small_moves():
    p = pd.Series([0.2, 0.6, 0.75, 0.5, 0.35, 0.25, 0.65])
    assert list(Al.band(p)) == [1, 1, 0, 0, 0, 1, 1]
    t = pd.Series([1.0, 0.95, 0.85, 0.84, 0.6])
    assert list(Al.sticky(t)) == [1.0, 1.0, 0.85, 0.85, 0.6]


def test_dummy_t_detects_a_shift_and_not_noise():
    from rh.reports import dummy_t

    rng = np.random.default_rng(0)
    z = (rng.random(4000) < 0.3).astype(float)
    assert abs(dummy_t(rng.normal(0, 1, 4000), z)) < 3
    assert dummy_t(rng.normal(0, 1, 4000) - 0.5 * z, z) < -8


def test_expanding_oos_only_fits_on_outcomes_known_by_the_forecast_day():
    import pandas as pd

    from rh.reports import expanding_oos

    idx = pd.bdate_range("2015-01-01", periods=900)
    x = np.sin(np.arange(900) / 20)
    y = 2 * x
    y[700:] = 2 * x[700:] + 5  # the relation shifts on day 700
    df = pd.DataFrame({"y": y, "x": x}, index=idx)
    e, _ = expanding_oos(df, ["x"], h=21, start=str(idx[600].date()), min_obs=100)
    k = 700 - 600  # error index of day 700
    assert np.allclose(e[: k + 21], np.r_[np.zeros(k), np.full(21, 5.0)], atol=1e-6)


def test_runs_and_toy_switch_timing():
    from rh.reports import runs, toy_switch

    idx = pd.bdate_range("2020-01-01", periods=6)
    flag = pd.Series([False, True, True, False, True, False], index=idx)
    r = runs(flag)
    assert list(r["days"]) == [2, 1] and r["start"].iloc[0] == idx[1]
    mkt = pd.Series(0.01, index=idx)
    rf = pd.Series(0.0002, index=idx)
    t = toy_switch(flag, mkt, rf)
    assert list(t) == [0.01, 0.01, 0.0002, 0.0002, 0.01, 0.0002]


def test_stats_compound_over_calendar_time():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    r = pd.Series(np.log(1.10) / len(idx), index=idx).pipe(np.expm1)
    s = Al.stats(r, pd.Series(0.0, index=idx))
    assert abs(s["cagr"] - 0.10) < 0.002
