"""Cleaning and feature construction, shared by the experiments and the final fit.

Everything here is label-free, so it can be applied identically to train,
validation and the December chart rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]
COORD_COLS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def daily_market_index(frames: list[pd.DataFrame]) -> pd.Series:
    """Mean market_index per date, from the rows of that date only.

    market_index is a market-wide daily level (std within a day ~0.025, across days
    ~0.17), so the per-row value is mostly the day's level plus noise. Averaging the
    ~150 loads of the same date also repairs the ~0.8% of rows where it is missing.
    It never looks across dates, so a fold cannot see another fold's future.
    """
    rows = pd.concat([f[["date", "market_index"]] for f in frames])
    return rows.groupby("date")["market_index"].mean()


def prepare(df: pd.DataFrame, mi_by_date: pd.Series, weight_fill: float) -> pd.DataFrame:
    out = df.copy()
    # |weight| runs 5,000-47,500 and the negatives mirror that range, so they are
    # sign errors rather than real values.
    out["weight"] = out["weight"].abs().fillna(weight_fill)
    out["mi_day"] = out["date"].map(mi_by_date)

    # Within a month the daily rate residual tracks market_index almost perfectly
    # (corr 0.95+), but each month also has its own level offset that the index does
    # not predict. So the transferable signal is the day's deviation from its month's
    # average level, not the absolute index. Months are complete in every file we score
    # (train Jan-Oct, validation Nov-Dec), so this needs no labels and no other fold.
    month_level = mi_by_date.groupby([mi_by_date.index.year, mi_by_date.index.month]).transform("mean")
    out["mi_dev"] = out["date"].map(mi_by_date - month_level)
    # Smoother reference (31-day centred window), kept only for a robustness check.
    out["mi_dev31"] = out["date"].map(mi_by_date - mi_by_date.rolling(31, center=True, min_periods=1).mean())
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_dist"] = np.log(out["distance"])
    for name in EQUIPMENT[1:]:
        out[f"eq_{name}"] = (out["equipment"] == name).astype(float)
    out["dow"] = out["date"].dt.dayofweek
    for k in range(1, 7):
        out[f"dow_{k}"] = (out["dow"] == k).astype(float)
    # Lets the market adjustment have a different slope per equipment type.
    out["mi_x_Reefer"] = out["mi_day"] * out["eq_Reefer"]
    out["mi_x_Flatbed"] = out["mi_day"] * out["eq_Flatbed"]
    return out


# Stage-1 inputs: nothing here knows the date, so the model cannot memorise single days.
STATIC = ["log_dist", "weight", "eq_Reefer", "eq_Flatbed", *COORD_COLS]
# Only used to flag corrupt labels (see models.clean_mask); never fed to the final model.
TRIM_FEATURES = ["log_dist", "weight", "eq_Reefer", "eq_Flatbed", "mi_day"]
