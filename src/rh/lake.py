"""NIFTY and India VIX from the local NSE archive (RH_LAKE_DIR, default
/path/to/lake), from 2013."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def lake_dir() -> Path:
    return Path(os.environ.get("RH_LAKE_DIR", Path.home() / "data" / "lake"))


def nifty_vix() -> pd.DataFrame:
    """Daily NIFTY log return in % and India VIX (annualised %, at that day's close)."""
    m = pd.read_parquet(lake_dir() / "market.parquet").set_index("date")[["nifty", "vix"]]
    m = m.dropna(subset=["nifty"]).sort_index()
    m["r"] = np.log(m["nifty"]).diff() * 100
    return m.dropna(subset=["r"])
