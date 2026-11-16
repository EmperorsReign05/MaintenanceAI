"""Feature engineering shared by training and the dashboard (one fitted object, saved to disk).

Steps (all statistics are fitted on TRAINING engines only):
  1. Drop non-informative sensors / settings whose training std is ~0.
  2. Multi-condition subsets (FD002/FD004): cluster operating settings into conditions (KMeans) and
     standardise every sensor within its condition, removing the operating-regime effect.
  3. Per engine, using only the current and PAST cycles (no look-ahead):
       rolling mean, rolling std and rolling linear slope over the last ROLLING_WINDOW cycles.
  4. StandardScaler over all engineered features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from . import config


class FeatureBuilder:
    def __init__(self, subset: str = config.DEFAULT_SUBSET, window: int = config.ROLLING_WINDOW):
        self.subset = subset
        self.window = window
        self.multi_condition = subset in config.MULTI_CONDITION_SUBSETS
        self.sensors: list[str] = []
        self.dropped: list[str] = []
        self.kmeans: KMeans | None = None
        self.cond_stats: dict[int, tuple[pd.Series, pd.Series]] = {}
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.sensor_ranges: dict[str, tuple[float, float]] = {}
        self.fitted = False

    # ----------------------------------------------------------------------- helpers
    @staticmethod
    def _check(df: pd.DataFrame) -> None:
        missing = [c for c in config.COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        vals = df[config.COLUMNS].apply(pd.to_numeric, errors="coerce")
        if vals.isna().any().any():
            raise ValueError(f"Non-numeric or empty values in: {vals.columns[vals.isna().any()].tolist()}")
        if (df["cycle"] < 1).any():
            raise ValueError("cycle must be >= 1")

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.multi_condition:
            return out
        cond = self.kmeans.predict(df[config.OP_SETTINGS].to_numpy())
        for c, (mu, sd) in self.cond_stats.items():
            m = cond == c
            if m.any():
                out.loc[m, self.sensors] = (df.loc[m, self.sensors] - mu) / sd
        return out

    def _rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["unit", "cycle"])
        g = df.groupby("unit", sort=False)
        w = self.window
        feats = {"cycle": df["cycle"].astype(float)}
        t = df["cycle"].astype(float)
        mt = g["cycle"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        mtt = (t * t).groupby(df["unit"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        var_t = (mtt - mt * mt).to_numpy()
        for s in self.sensors:
            x = df[s].astype(float)
            mx = g[s].transform(lambda v: v.rolling(w, min_periods=1).mean())
            sx = g[s].transform(lambda v: v.rolling(w, min_periods=2).std()).fillna(0.0)
            mxt = (x * t).groupby(df["unit"]).transform(lambda v: v.rolling(w, min_periods=1).mean())
            cov = (mxt - mx * mt).to_numpy()
            slope = np.divide(cov, var_t, out=np.zeros_like(cov), where=var_t > 1e-9)
            feats[f"{s}_raw"], feats[f"{s}_mean"], feats[f"{s}_std"], feats[f"{s}_slope"] = x, mx, sx, slope
        out = pd.DataFrame(feats, index=df.index)
        return out

    # --------------------------------------------------------------------------- API
    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self._check(train)
        if self.multi_condition:
            self.kmeans = KMeans(n_clusters=6, n_init=10, random_state=config.RANDOM_STATE)
            cond = self.kmeans.fit_predict(train[config.OP_SETTINGS].to_numpy())
            std_within = pd.concat([train.loc[cond == c, config.SENSORS].std() for c in range(6)], axis=1).max(axis=1)
            self.sensors = [s for s in config.SENSORS if std_within[s] > 1e-4]
            for c in range(6):
                part = train.loc[cond == c, self.sensors]
                self.cond_stats[c] = (part.mean(), part.std().replace(0, 1.0))
        else:
            std = train[config.SENSORS].std()
            self.sensors = [s for s in config.SENSORS if std[s] > 1e-4]
        self.dropped = [s for s in config.SENSORS if s not in self.sensors]
        self.sensor_ranges = {c: (float(train[c].min()), float(train[c].max()))
                              for c in [*config.OP_SETTINGS, *config.SENSORS]}
        X = self._rolling(self._normalise(train))
        self.feature_names = list(X.columns)
        self.scaler.fit(X.to_numpy())
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return scaled features aligned to df sorted by (unit, cycle). Index matches sorted df."""
        if not self.fitted:
            raise RuntimeError("FeatureBuilder must be fitted first")
        self._check(df)
        X = self._rolling(self._normalise(df))[self.feature_names]
        return pd.DataFrame(self.scaler.transform(X.to_numpy()), columns=self.feature_names, index=X.index)

    def out_of_range(self, reading: dict, tolerance: float = 0.05) -> list[str]:
        """Names of inputs outside the training range (+/- tolerance of the range width)."""
        bad = []
        for c, (lo, hi) in self.sensor_ranges.items():
            if c in reading and c not in self.dropped:
                pad = (hi - lo) * tolerance + 1e-9
                if not lo - pad <= float(reading[c]) <= hi + pad:
                    bad.append(c)
        return bad


def split_engines(train: pd.DataFrame, frac: float = config.VAL_ENGINE_FRACTION, seed: int = config.RANDOM_STATE):
    """Split by ENGINE so no engine's cycles appear in both sets (prevents temporal leakage)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(train, groups=train["unit"]))
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    assert set(tr.unit).isdisjoint(va.unit)
    return tr.sort_values(["unit", "cycle"]).reset_index(drop=True), va.sort_values(["unit", "cycle"]).reset_index(drop=True)
