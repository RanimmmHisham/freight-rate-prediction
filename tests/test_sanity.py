"""Cheap guards against the mistakes that would silently invalidate the results.

    python tests/test_sanity.py        (or: python -m pytest tests)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from harness import CORE, ROOT, mi, temporal_splits, train  # noqa: E402
from models import MarketAdjusted, to_dollars  # noqa: E402


def test_folds_are_strictly_in_the_future():
    for label, tr, te in temporal_splits():
        assert tr["date"].max() < te["date"].min(), label
        assert len(set(tr["load_id"]) & set(te["load_id"])) == 0


def test_inputs_are_complete_and_weight_is_sane():
    cols = CORE + ["mi_day", "mi_dev"]
    assert train[cols].notna().all().all()
    assert (train["weight"] > 0).all()
    assert mi.notna().all()


def test_market_deviation_is_centred_within_each_month():
    by_month = train.groupby(train["date"].dt.month).apply(lambda g: g.groupby("date")["mi_dev"].first().mean())
    assert by_month.abs().max() < 1e-9


def test_features_do_not_depend_on_the_label():
    # The validation file has no posted_rate, so the same code must give identical
    # features when the label column is removed from train.
    from features import add_features, prepare
    from harness import raw_train, weight_fill
    with_label = add_features(prepare(raw_train, mi, weight_fill))
    without = add_features(prepare(raw_train.drop(columns="posted_rate"), mi, weight_fill))
    pd.testing.assert_frame_equal(with_label.drop(columns="posted_rate"), without)


def test_model_is_deterministic():
    label, tr, te = next(temporal_splits())
    a = to_dollars(MarketAdjusted(CORE, ["mi_dev"]).fit(tr).predict(te), te)
    b = to_dollars(MarketAdjusted(CORE, ["mi_dev"]).fit(tr).predict(te), te)
    assert np.array_equal(a, b)


def test_submission_files_match_the_template():
    pred_path = ROOT / "validation_predictions.csv"
    if not pred_path.exists():
        return  # produced by src/train_predict.py
    pred = pd.read_csv(pred_path)
    template = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
    assert list(pred.columns) == ["load_id", "predicted_rate"]
    assert (pred["load_id"] == template["load_id"]).all()
    assert pred["predicted_rate"].notna().all() and (pred["predicted_rate"] > 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
