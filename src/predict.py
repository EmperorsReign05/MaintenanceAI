"""Inference using the saved preprocessing pipeline and the selected models."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config
from .insights import health_status
from .utils import load_json


class ArtifactsNotFoundError(FileNotFoundError):
    pass


class MaintenancePredictor:
    def __init__(self, rul_key: str | None = None, clf_key: str | None = None):
        if not config.METADATA_PATH.exists() or not config.FEATURE_BUILDER_PATH.exists():
            raise ArtifactsNotFoundError("No trained models found. Run: ./.venv/Scripts/python.exe train.py")
        self.meta = load_json(config.METADATA_PATH)
        self.rul_key = rul_key or self.meta["best_rul_model"]
        self.clf_key = clf_key or self.meta["best_clf_model"]
        for path in (config.reg_path(self.rul_key), config.clf_path(self.clf_key)):
            if not path.exists():
                raise ArtifactsNotFoundError(f"Missing {path.name}. Re-run: ./.venv/Scripts/python.exe train.py")
        self.builder = joblib.load(config.FEATURE_BUILDER_PATH)
        self.reg = joblib.load(config.reg_path(self.rul_key))
        self.clf = joblib.load(config.clf_path(self.clf_key))
        self.threshold = float(self.meta["classifiers"][self.clf_key]["threshold"])

    def predict_history(self, history: pd.DataFrame) -> pd.DataFrame:
        """Score every cycle of one or more engines' histories (columns: unit, cycle, op1-3, s1-s21)."""
        h = history.copy()
        if "unit" not in h.columns:
            h["unit"] = 0
        h = h.sort_values(["unit", "cycle"]).reset_index(drop=True)
        if h.duplicated(["unit", "cycle"]).any():
            raise ValueError("Duplicate (unit, cycle) rows in input history")
        X = self.builder.transform(h)
        rul = np.clip(self.reg.predict(X), 0, config.RUL_CAP)
        prob = self.clf.predict_proba(X)[:, 1]
        out = h[["unit", "cycle"]].copy()
        out["pred_rul"], out["fail_prob"] = rul, prob
        out["status"] = [health_status(r, p, self.threshold) for r, p in zip(rul, prob)]
        return out

    def predict_latest(self, history: pd.DataFrame) -> dict:
        res = self.predict_history(history).iloc[-1]
        return {"pred_rul": float(res.pred_rul), "fail_prob": float(res.fail_prob), "status": res.status}
