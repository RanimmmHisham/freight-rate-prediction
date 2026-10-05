"""The experiments behind every modelling decision. Writes CSVs to results/.

    python src/experiments.py          (about 10 minutes on 2 cores)

Validation is by time. The real task predicts the two months after the last training
date, so each fold trains on everything up to month k and scores month k+1 (Jul-Oct).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from harness import (CORE, OUT, BandMedian, GBM, GBMWithCityIds, MarketAdjusted, RobustLinear, cv, log_rpm, mi,
                     month, score, summarise, temporal_splits, to_dollars, train)

print(f"rows flagged as corrupt labels: {train['corrupt'].sum()} ({train['corrupt'].mean():.2%})")
DOW = [f"dow_{k}" for k in range(1, 7)]
FINAL = lambda: MarketAdjusted(CORE, ["mi_dev"])  # noqa: E731

# ---- E1: does the chosen model beat sensible baselines? ----------------------------
print("\n== E1 baselines vs models (temporal CV, mean of 4 folds) ==")
e1 = [
    cv("band median (equipment x distance band)", lambda: BandMedian()),
    cv("robust linear (distance, weight, equipment)", lambda: RobustLinear(["log_dist", "weight", "eq_Reefer", "eq_Flatbed"])),
    cv("GBM static features, raw labels", lambda: GBM(CORE)),
    cv("GBM static features, absolute-error loss", lambda: GBM(CORE, loss="absolute_error")),
    cv("GBM static features, corrupt labels trimmed", lambda: GBM(CORE, trim=True)),
    cv("final: trimmed GBM + month-deviation market adjustment", FINAL),
]
summarise(e1).to_csv(OUT / "e1_models.csv")
pd.concat(e1).to_csv(OUT / "e1_models_by_fold.csv", index=False)

# ---- E2: how should market_index be used? ------------------------------------------
print("\n== E2 market signal (trimmed labels throughout) ==")
e2 = [
    cv("no market input", lambda: GBM(CORE, trim=True)),
    cv("joint GBM + absolute market_index", lambda: GBM(CORE + ["mi_day"], trim=True)),
    cv("joint GBM + month-deviation", lambda: GBM(CORE + ["mi_dev"], trim=True)),
    cv("two-stage, absolute market_index", lambda: MarketAdjusted(CORE, ["mi_day"])),
    cv("two-stage, month-deviation (final)", FINAL),
    cv("two-stage, deviation from 31-day centred mean", lambda: MarketAdjusted(CORE, ["mi_dev31"])),
    cv("two-stage, month-deviation + day of week", lambda: MarketAdjusted(CORE, ["mi_dev"] + DOW)),
    cv("two-stage, month-deviation + quote_signal", lambda: MarketAdjusted(CORE, ["mi_dev", "quote_signal"])),
]
summarise(e2).to_csv(OUT / "e2_market_signal.csv")

print("\n== E2b which static inputs earn their place (trimmed GBM, no market input) ==")
e2b = [
    cv("all static inputs", lambda: GBM(CORE, trim=True)),
    cv("without weight", lambda: GBM([c for c in CORE if c != "weight"], trim=True)),
    cv("without coordinates", lambda: GBM(["log_dist", "weight", "eq_Reefer", "eq_Flatbed"], trim=True)),
    cv("without equipment", lambda: GBM([c for c in CORE if not c.startswith("eq_")], trim=True)),
]
summarise(e2b).to_csv(OUT / "e2b_static_inputs.csv")

# ---- E3: what the market signal looks like month by month, and the split trap -------
print("\n== E3a daily residual of the static model vs market_index, by month (out-of-fold) ==")
d = train[~train["corrupt"]].reset_index(drop=True)
oof = np.zeros(len(d))
for a, b in KFold(5, shuffle=True, random_state=0).split(d):
    oof[b] = GBM(CORE).fit(d.iloc[a]).predict(d.iloc[b])
d["res"] = log_rpm(d) - oof
day = d.groupby("date").agg(res=("res", "mean")).join(mi.rename("mi"))
rows = []
for m, g in day.groupby(day.index.month):
    slope, icpt = np.polyfit(g["mi"], g["res"], 1)
    rows.append({"month": m, "mean_market_index": g["mi"].mean(), "mean_residual": g["res"].mean(),
                 "within_month_corr": g["res"].corr(g["mi"]), "slope": slope, "intercept": icpt})
e3a = pd.DataFrame(rows).round(3)
print(e3a.to_string(index=False))
e3a.to_csv(OUT / "e3a_market_by_month.csv", index=False)

print("\n== E3b random 5-fold vs temporal: the same feature gets opposite verdicts ==")
rand = [(f"r{i}", train.iloc[a], train.iloc[b])
        for i, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(train))]
e3b = []
for split_name, splits in (("random 5-fold", rand), ("temporal", None)):
    for name, fs in (("static only", CORE), ("static + absolute market_index (joint GBM)", CORE + ["mi_day"])):
        e3b.append(cv(f"{split_name:13s} | {name}", lambda fs=fs: GBM(fs, trim=True), splits=splits))
summarise(e3b).to_csv(OUT / "e3b_split_comparison.csv")

# ---- E4: cities never seen in training ----------------------------------------------
print("\n== E4 hold out whole cities; score loads that touch them ==")
cities = np.array(sorted(set(train["pickup"]) | set(train["delivery"])))
groups = np.array_split(np.random.default_rng(0).permutation(cities), 5)
city_splits = []
for i, held in enumerate(groups):
    touches = train["pickup"].isin(held) | train["delivery"].isin(held)
    city_splits.append((f"city{i}", train[~touches], train[touches]))
e4 = [cv("coordinates only (final)", lambda: GBM(CORE, trim=True), splits=city_splits),
      cv("coordinates + city id", lambda: GBMWithCityIds(CORE, trim=True), splits=city_splits)]
summarise(e4).to_csv(OUT / "e4_unseen_cities.csv")
print("\n== E4b same comparison on the temporal folds (cities all seen) ==")
e4b = [cv("coordinates only", lambda: GBM(CORE, trim=True)),
       cv("coordinates + city id", lambda: GBMWithCityIds(CORE, trim=True))]
summarise(e4b).to_csv(OUT / "e4b_city_id_temporal.csv")

# ---- E5: hyper-parameters (static model; the adjustment has no tunables) ------------
print("\n== E5 hyper-parameters ==")
grid = [dict(max_leaf_nodes=l, min_samples_leaf=s, learning_rate=lr, max_iter=it)
        for l in (7, 15, 31) for s in (40, 200) for lr, it in ((0.06, 300), (0.03, 600))]
e5 = []
for p in grid:
    tag = f"leaves={p['max_leaf_nodes']} min_leaf={p['min_samples_leaf']} lr={p['learning_rate']} iters={p['max_iter']}"
    e5.append(cv(tag, lambda p=p: GBM(CORE, trim=True, **p)))
e5s = summarise(e5)
e5s.to_csv(OUT / "e5_hyperparameters.csv")
print("spread of clean MAPE across the grid:", round(e5s["clean_MAPE"].min(), 4), "to", round(e5s["clean_MAPE"].max(), 4))

# ---- E6: does scaling the forecast help? (labels contain corrupt rows) --------------
print("\n== E6 multiplicative calibration of the final forecast ==")
rows = []
for label, tr, te in temporal_splits():
    base = to_dollars(FINAL().fit(tr).predict(te), te)
    for c in (0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.05):
        rows.append(score(base * c, te, label, {"scale": c}))
e6 = pd.DataFrame(rows).groupby("scale").mean(numeric_only=True).round(4)
print(e6[["all_MAE", "all_RMSE", "all_MAPE", "all_MedAPE", "clean_MAE"]])
e6.to_csv(OUT / "e6_calibration.csv")

# ---- E7: where does the final model fail? (October fold) ---------------------------
print("\n== E7 error analysis, October fold (train Jan-Sep) ==")
tr, te = train[month <= 9], train[month == 10].copy()
te["pred"] = to_dollars(FINAL().fit(tr).predict(te), te)
te["ape"] = (te["pred"] - te["posted_rate"]).abs() / te["posted_rate"]
seen = set(zip(tr["pickup"], tr["delivery"]))
te["lane_seen"] = [(a, b) in seen for a, b in zip(te["pickup"], te["delivery"])]
te["dist_band"] = pd.cut(te["distance"], [0, 300, 600, 1000, 1500, 2500, 4000])
slices = {
    "all rows": te.groupby(lambda _: "all")["ape"],
    "label corrupt vs clean": te.groupby(te["corrupt"].map({True: "corrupt", False: "clean"}))["ape"],
    "distance band": te.groupby("dist_band", observed=True)["ape"],
    "equipment": te.groupby("equipment")["ape"],
    "lane seen in training": te.groupby("lane_seen")["ape"],
}
parts = []
for name, g in slices.items():
    s = g.agg(n="size", MAPE="mean", MedAPE="median").round(4)
    print(name); print(s)
    s = s.reset_index()
    s.columns = ["group", *s.columns[1:]]
    s.insert(0, "slice", name)
    parts.append(s)
pd.concat(parts, ignore_index=True).to_csv(OUT / "e7_error_analysis.csv", index=False)
