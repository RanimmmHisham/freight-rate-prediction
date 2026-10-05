"""Shared data loading, temporal splits and scoring for the experiment scripts."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import STATIC, TRIM_FEATURES, add_features, daily_market_index, load, prepare  # noqa: F401
from models import (GBM, BandMedian, GBMWithCityIds, MarketAdjusted, RobustLinear, clean_mask,  # noqa: F401
                    log_rpm, metrics, to_dollars)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

raw_train = load(ROOT / "data/train_test.csv")
raw_val = load(ROOT / "data/validation.csv")
mi = daily_market_index([raw_train, raw_val])
weight_fill = float(raw_train["weight"].abs().median())
train = add_features(prepare(raw_train, mi, weight_fill)).reset_index(drop=True)

train["corrupt"] = ~clean_mask(train)

month = train["date"].dt.month
FOLDS = [(m - 1, m) for m in (7, 8, 9, 10)]  # (last train month, test month)
CORE = STATIC


def temporal_splits(frame=None):
    frame = train if frame is None else frame
    m = frame["date"].dt.month
    for last_train, test_month in FOLDS:
        yield f"{test_month:02d}", frame[m <= last_train], frame[m == test_month]


def score(pred_dollars, test, label, extra=None):
    y = test["posted_rate"].to_numpy()
    row = {**(extra or {}), "fold": label}
    row.update({f"all_{k}": v for k, v in metrics(y, pred_dollars).items()})
    clean = ~test["corrupt"].to_numpy()
    row.update({f"clean_{k}": v for k, v in metrics(y[clean], pred_dollars[clean]).items()})
    return row


def cv(name, make_model, splits=None, quiet=False):
    rows = []
    for label, tr, te in (splits or temporal_splits()):
        t0 = time.time()
        model = make_model().fit(tr)
        pred = to_dollars(model.predict(te), te)
        r = score(pred, te, label, {"model": name})
        r["fit_s"] = round(time.time() - t0, 1)
        r["n_trimmed"] = getattr(model, "n_trimmed", 0)
        rows.append(r)
    df = pd.DataFrame(rows)
    if not quiet:
        mean = df.drop(columns=["fold", "model"]).mean()
        print(f"{name:55s} MAE {mean['all_MAE']:8.1f}  MedAPE {mean['all_MedAPE']:.4f}  "
              f"| clean MAE {mean['clean_MAE']:7.1f}  clean MAPE {mean['clean_MAPE']:.4f}")
    return df


def summarise(frames):
    return pd.concat(frames).groupby("model", sort=False).mean(numeric_only=True).round(4)
