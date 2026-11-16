"""Unit tests. A tiny randomly generated frame with the C-MAPSS COLUMN LAYOUT is used only to exercise
code paths; it is never used to train the real models.   Run:  pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import DatasetNotFoundError, add_test_rul, add_train_rul, dataset_paths, label_failure, read_cmapss
from src.evaluate import classification_metrics, nasa_score, regression_metrics
from src.features import FeatureBuilder, split_engines
from src.insights import grouped_permutation_importance, health_status, recommend
from src.models import build_classifier, build_regressor


def make_engines(n_units=12, seed=0, multi=False):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = int(rng.integers(60, 120))
        for c in range(1, life + 1):
            deg = np.exp(-(life - c) / 25)
            ops = rng.choice([0.0, 20.0, 42.0]) if multi else 0.0
            s = 100 + np.arange(21) + deg * 5 * np.sign(np.arange(21) % 2 - 0.5) + rng.normal(0, 0.05, 21) + ops
            s[[0, 4, 5]] = 518.67 + ops     # constant within condition
            rows.append([u, c, ops, ops / 100, 100.0, *s])
    return pd.DataFrame(rows, columns=config.COLUMNS)


@pytest.fixture(scope="module")
def train():
    return add_train_rul(make_engines())


def test_missing_files(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        dataset_paths("FD001", tmp_path)


def test_read_cmapss_format(tmp_path):
    df = make_engines(2)
    p = tmp_path / "train_FD001.txt"
    p.write_text("".join(" ".join(f"{v:.4f}" for v in r) + " \n" for r in df.to_numpy()))
    back = read_cmapss(p)
    assert list(back.columns) == config.COLUMNS and len(back) == len(df)
    assert back["unit"].dtype.kind == "i"


def test_rul_labels(train):
    last = train.groupby("unit").tail(1)
    assert (last.rul_true == 0).all()
    assert train.rul.max() <= config.RUL_CAP
    t = make_engines(2).groupby("unit").head(30)
    t = add_test_rul(t, pd.Series([10, 50]))
    assert t.groupby("unit").tail(1).rul_true.tolist() == [10, 50]
    assert t[t.unit == 1].rul_true.iloc[0] == 10 + 29
    assert label_failure([10, 30, 31]).tolist() == [1, 1, 0]


def test_engine_split_has_no_overlap(train):
    tr, va = split_engines(train)
    assert set(tr.unit).isdisjoint(va.unit)
    assert len(tr) + len(va) == len(train)


def test_features_are_causal(train):
    tr, _ = split_engines(train)
    fb = FeatureBuilder().fit(tr)
    assert {"s1", "s5", "s6"}.issubset(fb.dropped)
    full = tr[tr.unit == tr.unit.iloc[0]].reset_index(drop=True)
    X_full = fb.transform(full)
    X_cut = fb.transform(full.head(20))
    # features at cycle 20 must not change when future cycles are added
    np.testing.assert_allclose(X_full.iloc[19].to_numpy(), X_cut.iloc[19].to_numpy(), atol=1e-9)


def test_single_reading_and_validation(train):
    fb = FeatureBuilder().fit(train)
    one = train.head(1)
    X = fb.transform(one)
    assert X.shape == (1, len(fb.feature_names)) and np.isfinite(X.to_numpy()).all()
    with pytest.raises(ValueError):
        fb.transform(one.drop(columns=["s2"]))
    assert fb.out_of_range({"s2": 1e6}) == ["s2"]


def test_multi_condition_normalisation():
    tr = add_train_rul(make_engines(multi=True, seed=3))
    fb = FeatureBuilder("FD002").fit(tr)
    X = fb.transform(tr)
    assert np.isfinite(X.to_numpy()).all()


@pytest.mark.parametrize("key", ["linear_regression", "random_forest", "svr"])
def test_regressors(train, key):
    tr, va = split_engines(train)
    fb = FeatureBuilder().fit(tr)
    m = build_regressor(key).fit(fb.transform(tr), tr.rul)
    p = m.predict(fb.transform(va))
    met = regression_metrics(va.rul, p)
    assert set(met) >= {"mae", "rmse", "r2"} and met["rmse"] > 0


@pytest.mark.parametrize("key", ["logistic_regression", "random_forest"])
def test_classifiers(train, key):
    fb = FeatureBuilder().fit(train)
    X, y = fb.transform(train), label_failure(train.rul_true)
    s = build_classifier(key).fit(X, y).predict_proba(X)[:, 1]
    assert classification_metrics(y, s, 0.5)["roc_auc"] > 0.8


def test_metric_definitions():
    assert nasa_score([10], [10]) == 0
    assert nasa_score([10], [20]) > nasa_score([20], [10])   # late is penalised more than early
    m = classification_metrics(np.array([0, 1, 1, 0]), np.array([0.2, 0.9, 0.3, 0.7]), 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_status_and_recommendation():
    assert health_status(10, 0.1, 0.5) == "High risk"
    assert health_status(100, 0.9, 0.5) == "High risk"
    assert health_status(60, 0.1, 0.5) == "Warning"
    assert health_status(110, 0.1, 0.5) == "Healthy"
    rec = recommend("High risk", 12, 0.8, ["s11"])
    assert "NOT an engineering diagnosis" in rec["disclaimer"] and len(rec["actions"]) >= 3


def test_permutation_importance(train):
    fb = FeatureBuilder().fit(train)
    X = fb.transform(train)
    m = build_regressor("linear_regression").fit(X, train.rul)
    imp = grouped_permutation_importance(m, X, train.rul, fb.sensors, n_repeats=1)
    assert len(imp) == len(fb.sensors) + 1 and imp.importance.iloc[0] >= imp.importance.iloc[-1]
