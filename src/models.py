"""Models and metrics. Every model predicts log(rate per mile) and returns dollars."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from features import TRIM_FEATURES

HGB_DEFAULTS = dict(learning_rate=0.06, max_iter=300, max_leaf_nodes=7,
                    min_samples_leaf=40, l2_regularization=1.0, random_state=0)
TRIM_SIGMAS = 4.0


def log_rpm(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["posted_rate"].to_numpy() / df["distance"].to_numpy())


def to_dollars(pred_log_rpm: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    return np.exp(pred_log_rpm) * df["distance"].to_numpy()


# baseline 0: median rate per mile in (equipment x distance band)
class BandMedian:
    def __init__(self, n_bands: int = 12):
        self.n_bands = n_bands

    def fit(self, df):
        self.edges = np.unique(np.quantile(df["distance"], np.linspace(0, 1, self.n_bands + 1)))
        band = np.clip(np.searchsorted(self.edges, df["distance"], side="right") - 1, 0, len(self.edges) - 2)
        key = pd.DataFrame({"eq": df["equipment"].to_numpy(), "band": band, "y": log_rpm(df)})
        self.table = key.groupby(["eq", "band"])["y"].median()
        self.fallback = float(np.median(key["y"]))
        return self

    def predict(self, df):
        band = np.clip(np.searchsorted(self.edges, df["distance"], side="right") - 1, 0, len(self.edges) - 2)
        idx = pd.MultiIndex.from_arrays([df["equipment"].to_numpy(), band])
        return self.table.reindex(idx).fillna(self.fallback).to_numpy()


# baseline 1: robust linear model in log space
class RobustLinear:
    def __init__(self, features):
        self.features = features

    def _x(self, df):
        x = df[self.features].copy()
        x["log_dist_sq"] = df["log_dist"] ** 2
        return x

    def fit(self, df):
        self.scaler = StandardScaler().fit(self._x(df))
        # Huber loss so the ~1.4% corrupt labels do not drag the line.
        self.model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500).fit(
            self.scaler.transform(self._x(df)), log_rpm(df))
        return self

    def predict(self, df):
        return self.model.predict(self.scaler.transform(self._x(df)))


# gradient boosting
class GBM:
    def __init__(self, features, loss="squared_error", trim=False, categorical=None, **params):
        self.features, self.loss, self.trim = features, loss, trim
        self.categorical = categorical or []
        self.params = {**HGB_DEFAULTS, **params}

    def fit(self, df):
        if self.trim:
            keep = clean_mask(df)
            df = df[keep]
            self.n_trimmed = int((~keep).sum())
        cat_mask = [c in self.categorical for c in self.features]
        self.model = HistGradientBoostingRegressor(
            loss=self.loss, categorical_features=cat_mask if any(cat_mask) else None, **self.params)
        self.model.fit(df[self.features], log_rpm(df))
        return self

    def predict(self, df):
        return self.model.predict(df[self.features])


class GBMWithCityIds(GBM):
    """GBM plus pickup/delivery city as categorical features. Unseen cities become missing."""

    def __init__(self, base_features, **kw):
        super().__init__(list(base_features) + ["pickup_id", "delivery_id"],
                         categorical=["pickup_id", "delivery_id"], **kw)

    def _encode(self, df, fit=False):
        if fit:
            self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
            self.enc.fit(pd.concat([df["pickup"], df["delivery"]]).to_frame("c"))
        out = df.copy()
        for c in ("pickup", "delivery"):
            out[c + "_id"] = self.enc.transform(df[[c]].rename(columns={c: "c"})).ravel()
        return out

    def fit(self, df):
        return super().fit(self._encode(df, fit=True))

    def predict(self, df):
        return super().predict(self._encode(df))


class MarketAdjusted:
    """Static-feature GBM plus a small linear correction for the day's market level.

    Stage 1 never sees the date or market_index, so it cannot memorise day-specific
    shocks. Stage 2 regresses stage 1's *out-of-fold* log residuals on a handful of
    same-day inputs. Out-of-fold matters: in-sample residuals are too small and would
    make the correction look weaker than it is.
    """

    def __init__(self, static_features, adj_features=("mi_day",), **gbm_params):
        self.static, self.adj, self.gbm_params = list(static_features), list(adj_features), gbm_params

    def fit(self, df):
        keep = clean_mask(df)
        self.n_trimmed = int((~keep).sum())
        d = df[keep]
        y = log_rpm(d)
        oof = np.zeros(len(d))
        for a, b in KFold(5, shuffle=True, random_state=0).split(d):
            oof[b] = GBM(self.static, **self.gbm_params).fit(d.iloc[a]).predict(d.iloc[b])
        self.scaler = StandardScaler().fit(d[self.adj])
        self.stage2 = HuberRegressor(epsilon=1.35, alpha=1e-6).fit(self.scaler.transform(d[self.adj]), y - oof)
        self.stage1 = GBM(self.static, **self.gbm_params).fit(d)
        return self

    def predict(self, df):
        return self.stage1.predict(df) + self.stage2.predict(self.scaler.transform(df[self.adj]))


def clean_mask(train: pd.DataFrame) -> np.ndarray:
    """Rows whose label sits within TRIM_SIGMAS robust sigmas of a robust linear fit.

    Fitted on the training rows only, so it can be used inside a CV fold without
    touching the held-out period.
    """
    lin = RobustLinear(TRIM_FEATURES).fit(train)
    resid = log_rpm(train) - lin.predict(train)
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return np.abs(resid) < TRIM_SIGMAS * sigma


# metrics
def metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(ape)),
        "MedAPE": float(np.median(ape)),
        "WAPE": float(np.sum(np.abs(err)) / np.sum(y_true)),
    }
