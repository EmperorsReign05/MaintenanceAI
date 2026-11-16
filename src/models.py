"""Model builders and hyper-parameter search spaces.

RUL regression (target = capped RUL in cycles):
  * Linear Regression  - baseline; assumes RUL is a weighted sum of features.
  * Random Forest      - bagged decision trees; captures non-linear degradation, robust to noise.
  * XGBoost            - gradient-boosted trees; each tree corrects the previous ones' errors.
  * SVR (RBF kernel)   - fits a function within an epsilon-tube; strong on smooth non-linear trends,
                         but O(n^2) so trained on a fixed subsample.
Failure-risk classification (target = RUL <= FAILURE_WINDOW):
  * Logistic Regression (class-weighted), Random Forest (balanced_subsample), XGBoost (scale_pos_weight).
"""
from __future__ import annotations

from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR

from . import config
from .utils import get_logger, is_available

log = get_logger()
HAS_XGB = is_available("xgboost")
SEED = config.RANDOM_STATE


def regressor_keys() -> list[str]:
    keys = ["linear_regression", "random_forest"]
    if HAS_XGB:
        keys.append("xgboost")
    else:
        log.warning("xgboost not installed -> XGBoost skipped (pip install xgboost)")
    return keys + ["svr"]


def classifier_keys() -> list[str]:
    return ["logistic_regression", "random_forest"] + (["xgboost"] if HAS_XGB else [])


def build_regressor(key: str):
    if key == "linear_regression":
        return LinearRegression()
    if key == "random_forest":
        return RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                     max_features=0.5, n_jobs=-1, random_state=SEED)
    if key == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                            n_jobs=-1, random_state=SEED)
    if key == "svr":
        return SVR(kernel="rbf", C=30.0, epsilon=2.0, gamma="scale")
    raise ValueError(f"Unknown regressor {key}")


def build_classifier(key: str, scale_pos_weight: float = 1.0):
    if key == "logistic_regression":
        return LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)
    if key == "random_forest":
        return RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=5,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    if key == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                             eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED)
    raise ValueError(f"Unknown classifier {key}")


def regressor_space(key: str) -> dict:
    return {
        "random_forest": {"n_estimators": [150, 250, 350], "max_depth": [8, 12, 16, None],
                          "min_samples_leaf": [2, 5, 10], "max_features": ["sqrt", 0.3, 0.5]},
        "xgboost": {"n_estimators": [200, 400, 700], "max_depth": [3, 4, 5, 6],
                    "learning_rate": [0.02, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10]},
        "svr": {"C": loguniform(1, 200), "epsilon": [0.5, 2.0, 5.0], "gamma": ["scale", 0.01, 0.03]},
    }.get(key, {})


def classifier_space(key: str) -> dict:
    return {
        "logistic_regression": {"C": loguniform(1e-3, 10)},
        "random_forest": {"n_estimators": [150, 250, 350], "max_depth": [8, 12, None],
                          "min_samples_leaf": [2, 5, 10]},
        "xgboost": {"n_estimators": [200, 400], "max_depth": [3, 4, 6], "learning_rate": [0.03, 0.05, 0.1],
                    "subsample": [0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0]},
    }.get(key, {})
