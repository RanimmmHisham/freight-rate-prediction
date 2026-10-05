"""Prints the data-quality numbers quoted in the report and the Loom.

    python src/data_checks.py            (also saved to results/data_checks.txt)
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from features import haversine_miles
from harness import ROOT, OUT, month, raw_train, raw_val, train

dec = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
buf = io.StringIO()
with redirect_stdout(buf):
    print("== shape and time ==")
    print(f"train {raw_train.shape}, {raw_train['date'].min().date()} to {raw_train['date'].max().date()}")
    print(f"validation {raw_val.shape}, {raw_val['date'].min().date()} to {raw_val['date'].max().date()}")
    print(f"duplicate load_id in train: {raw_train['load_id'].duplicated().sum()}")

    print("\n== missing values ==")
    for name, df in (("train", raw_train), ("validation", raw_val)):
        print(name, df.isna().sum()[lambda s: s > 0].to_dict())
    print("December chart inputs columns:", list(dec.columns))

    print("\n== weight ==")
    for name, df in (("train", raw_train), ("validation", raw_val)):
        w = df["weight"]
        print(f"{name}: negative {int((w < 0).sum())}, |weight| min {w.abs().min():.0f} max {w.abs().max():.0f}, "
              f"share at exactly 47,500 {(w.abs() == 47500).mean():.2%}")

    print("\n== labels ==")
    rpm = raw_train["posted_rate"] / raw_train["distance"]
    print(f"rate per mile: median {rpm.median():.2f}, 1st pct {rpm.quantile(.01):.2f}, 99th {rpm.quantile(.99):.2f}, "
          f"max {rpm.max():.2f}, min {rpm.min():.2f}")
    bad = train[train["corrupt"]]
    ratio = bad["posted_rate"] / bad["posted_rate"].div(bad["distance"]).groupby(bad["equipment"]).transform("median") / bad["distance"]
    print(f"labels flagged as corrupt (> 4 robust sigma from a robust fit): {len(bad)} ({len(bad) / len(train):.2%}); "
          f"{(ratio > 1).sum()} too high, {(ratio < 1).sum()} too low")
    print("share flagged by month:", (train.groupby(month)["corrupt"].mean().round(4)).to_dict())

    print("\n== distance vs coordinates ==")
    hav = haversine_miles(raw_train["pickup_lat"], raw_train["pickup_lon"], raw_train["delivery_lat"], raw_train["delivery_lon"])
    r = raw_train["distance"] / hav
    print(f"road/straight-line ratio: median {r.median():.3f}, 99th pct {r.quantile(.99):.3f}, "
          f"rows above 1.6: {(r > 1.6).sum()} ({(r > 1.6).mean():.2%}); corr(distance, haversine) = {raw_train['distance'].corr(hav):.4f}")

    print("\n== validation coverage ==")
    seen_cities = set(raw_train["pickup"]) | set(raw_train["delivery"])
    unseen = (set(raw_val["pickup"]) | set(raw_val["delivery"])) - seen_cities
    touch = raw_val["pickup"].isin(unseen) | raw_val["delivery"].isin(unseen)
    lanes = set(zip(raw_train["pickup"], raw_train["delivery"]))
    lane_unseen = [(a, b) not in lanes for a, b in zip(raw_val["pickup"], raw_val["delivery"])]
    print(f"cities in validation but not train: {sorted(unseen)}; loads touching them: {touch.mean():.1%}")
    print(f"loads on a lane never seen in train: {np.mean(lane_unseen):.1%}")
    print(f"December lane Lexington -> Fort Wayne in train: "
          f"{((raw_train.pickup == 'Lexington') & (raw_train.delivery == 'Fort Wayne')).sum()} loads")

    print("\n== market_index and quote_signal ==")
    by_day = raw_train.groupby("date")["market_index"]
    print(f"market_index: std within a day {by_day.std().mean():.3f}, std of daily means {by_day.mean().std():.3f}, "
          f"day-to-day autocorrelation {by_day.mean().autocorr(1):.2f}")
    print(f"corr(rate per mile, market_index) {rpm.corr(raw_train['market_index']):.3f}; "
          f"corr(rate per mile, quote_signal) {rpm.corr(raw_train['quote_signal']):.3f}")
    q = raw_train.groupby("date")["quote_signal"]
    print(f"quote_signal: std within a day {q.std().mean():.3f}, std of daily means {q.mean().std():.3f}")

text = buf.getvalue()
print(text)
(OUT / "data_checks.txt").write_text(text)
