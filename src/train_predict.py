"""Fit the final model on all labelled data, then write the two prediction files.

    python src/train_predict.py
    python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import STATIC, add_features, daily_market_index, load, prepare
from models import MarketAdjusted, to_dollars

ROOT = Path(__file__).resolve().parent.parent
raw_train = load(ROOT / "data/train_test.csv")
raw_val = load(ROOT / "data/validation.csv")
template = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
dec = pd.read_csv(ROOT / "data/december_chart_inputs.csv", parse_dates=["date"])

mi = daily_market_index([raw_train, raw_val])
weight_fill = float(raw_train["weight"].abs().median())

coords = pd.concat([
    raw_train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    raw_train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
]).drop_duplicates("city").set_index("city")
for side in ("pickup", "delivery"):
    dec[f"{side}_lat"] = dec[side].map(coords["lat"])
    dec[f"{side}_lon"] = dec[side].map(coords["lon"])
assert dec[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all().all()

train = add_features(prepare(raw_train, mi, weight_fill))
val = add_features(prepare(raw_val, mi, weight_fill))
dec_f = add_features(prepare(dec, mi, weight_fill))
assert val[STATIC + ["mi_dev"]].notna().all().all() and dec_f[STATIC + ["mi_dev"]].notna().all().all()

model = MarketAdjusted(STATIC, ["mi_dev"]).fit(train)
print(f"trained on {len(train) - model.n_trimmed:,} rows; {model.n_trimmed} corrupt-looking labels set aside")
print("market adjustment (log-rate change per +1 sd of deviation):", model.stage2.coef_.round(4))

pred = pd.Series(to_dollars(model.predict(val), val), index=val["load_id"])
out = template[["load_id"]].copy()
out["predicted_rate"] = out["load_id"].map(pred).round(2)
assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
out.to_csv(ROOT / "validation_predictions.csv", index=False)

dec_out = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
dec_out["predicted_rate"] = np.round(to_dollars(model.predict(dec_f), dec_f), 2)
dec_out.to_csv(ROOT / "data/december_chart_inputs_predicted.csv", index=False)

print(f"\nvalidation: {len(out):,} rows, predicted rate mean ${out['predicted_rate'].mean():,.0f}, "
      f"range ${out['predicted_rate'].min():,.0f} to ${out['predicted_rate'].max():,.0f}")
print(f"December lane: ${dec_out['predicted_rate'].min():,.0f} to ${dec_out['predicted_rate'].max():,.0f}, "
      f"mean ${dec_out['predicted_rate'].mean():,.0f}")
