"""Predictive Maintenance AI - training pipeline (CPU only).

    .venv\\Scripts\\python.exe train.py             # FD001, tuned (Windows)
    .venv\\Scripts\\python.exe train.py --no-tune   # fast run with defaults
    .venv\\Scripts\\python.exe train.py --subset FD003
    .venv\\Scripts\\python.exe train.py --lstm      # needs tensorflow
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from src import config
from src.data_loader import DatasetNotFoundError, label_failure, load_subset, missing_report
from src.eda import run_eda
from src.evaluate import (
    best_f1_threshold, classification_metrics, plot_confusion, plot_importance, plot_roc, plot_rul_bars,
    plot_rul_scatter, regression_metrics, roc_points,
)
from src.features import FeatureBuilder, split_engines
from src.insights import grouped_permutation_importance, health_status
from src.models import (
    build_classifier, build_regressor, classifier_keys, classifier_space, regressor_keys, regressor_space,
)
from src.utils import get_logger, save_json

log = get_logger()


def parse_args():
    p = argparse.ArgumentParser(description="Train Predictive Maintenance AI")
    p.add_argument("--subset", default=config.DEFAULT_SUBSET, choices=config.SUBSETS)
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--no-tune", action="store_true", help="skip hyper-parameter search")
    p.add_argument("--n-iter", type=int, default=8, help="random-search iterations per model")
    p.add_argument("--cv", type=int, default=3, help="GroupKFold folds (grouped by engine)")
    p.add_argument("--lstm", action="store_true", help="also train the optional LSTM")
    p.add_argument("--lstm-epochs", type=int, default=40)
    p.add_argument("--skip-eda", action="store_true")
    return p.parse_args()


def search(model, space, X, y, groups, args, scoring):
    if args.no_tune or not space:
        model.fit(X, y)
        return model, {}
    s = RandomizedSearchCV(model, space, n_iter=args.n_iter, scoring=scoring, cv=GroupKFold(n_splits=args.cv),
                           random_state=config.RANDOM_STATE, n_jobs=1, refit=True, error_score="raise")
    s.fit(X, y, groups=groups)
    log.info("   best CV %s %.4f  %s", scoring, s.best_score_, s.best_params_)
    params = {k: (v.item() if hasattr(v, "item") else v) for k, v in s.best_params_.items()}
    return s.best_estimator_, params


def last_cycle_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["cycle"] == df.groupby("unit")["cycle"].transform("max")).to_numpy()


def main() -> int:
    args = parse_args()
    config.set_seeds()
    config.ensure_dirs()
    t0 = time.time()

    # 1. Load ----------------------------------------------------------------------------
    try:
        train_full, test = load_subset(args.subset, args.data_dir)
    except DatasetNotFoundError as exc:
        print(exc)
        return 1
    miss = {"train": missing_report(train_full), "test": missing_report(test)}

    # 2. Engine-wise split + preprocessing fitted on training engines only -----------------
    tr, va = split_engines(train_full)
    builder = FeatureBuilder(args.subset).fit(tr)
    joblib.dump(builder, config.FEATURE_BUILDER_PATH)
    log.info("Engines: train %d | validation %d | test %d. Informative sensors (%d): %s",
             tr.unit.nunique(), va.unit.nunique(), test.unit.nunique(), len(builder.sensors),
             ", ".join(builder.sensors))
    X_tr, X_va, X_te = builder.transform(tr), builder.transform(va), builder.transform(test)
    g_tr = tr["unit"].to_numpy()
    te_last = last_cycle_mask(test)
    y_te_last = test.loc[te_last, "rul_true"].to_numpy()

    eda = None if args.skip_eda else run_eda(train_full, builder.sensors, args.subset)

    # 3. RUL regression ----------------------------------------------------------------------
    reg_meta, reg_rows, reg_metrics, last_preds = {}, [], {}, {}
    rng = np.random.default_rng(config.RANDOM_STATE)
    for key in regressor_keys():
        name = config.REG_NAMES[key]
        log.info("RUL model: %s", name)
        t = time.time()
        Xf, yf, gf = X_tr, tr["rul"].to_numpy(), g_tr
        if key == "svr" and len(Xf) > config.SVR_MAX_ROWS:
            idx = np.sort(rng.choice(len(Xf), config.SVR_MAX_ROWS, replace=False))
            Xf, yf, gf = Xf.iloc[idx], yf[idx], gf[idx]
            log.info("   SVR trained on a %d-row random subsample", config.SVR_MAX_ROWS)
        try:
            model, params = search(build_regressor(key), regressor_space(key), Xf, yf, gf, args,
                                   "neg_root_mean_squared_error")
        except Exception as exc:
            log.error("   %s failed: %s", name, exc)
            continue
        secs = time.time() - t
        p_va = np.clip(model.predict(X_va), 0, config.RUL_CAP)
        p_te = np.clip(model.predict(X_te), 0, config.RUL_CAP)
        m = {"validation": regression_metrics(va["rul"], p_va),
             "test_last_cycle": regression_metrics(y_te_last, p_te[te_last]),
             "test_all_cycles": regression_metrics(test["rul"], p_te)}
        reg_metrics[name] = m
        last_preds[key] = p_te[te_last]
        joblib.dump(model, config.reg_path(key))
        reg_meta[key] = {"label": name, "params": params, "val_rmse": m["validation"]["rmse"],
                         "train_seconds": round(secs, 1)}
        tm = m["test_last_cycle"]
        reg_rows.append({"model_key": key, "model": name, **{k: tm[k] for k in ("mae", "rmse", "r2", "nasa_score")},
                         "val_rmse": m["validation"]["rmse"], "test_all_rmse": m["test_all_cycles"]["rmse"],
                         "train_seconds": round(secs, 1), "deployable": True})
        log.info("   TEST(last cycle) RMSE %.2f | MAE %.2f | R2 %.3f  (%.0fs)", tm["rmse"], tm["mae"], tm["r2"], secs)

    if not reg_meta:
        log.error("No RUL model trained.")
        return 1
    best_rul = min(reg_meta, key=lambda k: reg_meta[k]["val_rmse"])

    # Optional LSTM (compared, not used for live dashboard predictions)
    if args.lstm:
        from src.lstm import HAS_TF, make_sequences, predict_lstm, sequence_columns, train_lstm
        if not HAS_TF:
            log.warning("--lstm requested but TensorFlow is not installed (pip install tensorflow)")
        else:
            log.info("RUL model: LSTM (sequence length %d)", config.LSTM_SEQUENCE)
            cols = sequence_columns(builder.feature_names)
            S_tr = make_sequences(X_tr[cols], g_tr)
            S_va = make_sequences(X_va[cols], va["unit"].to_numpy())
            S_te = make_sequences(X_te[cols], test["unit"].to_numpy())
            t = time.time()
            net = train_lstm(S_tr, tr["rul"], S_va, va["rul"], args.lstm_epochs)
            secs = time.time() - t
            p_va, p_te = predict_lstm(net, S_va), predict_lstm(net, S_te)
            m = {"validation": regression_metrics(va["rul"], p_va),
                 "test_last_cycle": regression_metrics(y_te_last, p_te[te_last]),
                 "test_all_cycles": regression_metrics(test["rul"], p_te)}
            reg_metrics["LSTM"] = m
            last_preds["lstm"] = p_te[te_last]
            tm = m["test_last_cycle"]
            reg_rows.append({"model_key": "lstm", "model": "LSTM",
                             **{k: tm[k] for k in ("mae", "rmse", "r2", "nasa_score")},
                             "val_rmse": m["validation"]["rmse"], "test_all_rmse": m["test_all_cycles"]["rmse"],
                             "train_seconds": round(secs, 1), "deployable": False})
            log.info("   TEST(last cycle) RMSE %.2f | MAE %.2f | R2 %.3f", tm["rmse"], tm["mae"], tm["r2"])

    # 4. Failure-risk classification ---------------------------------------------------------
    y_tr_c, y_va_c = label_failure(tr["rul_true"]), label_failure(va["rul_true"])
    y_te_c = label_failure(test["rul_true"])
    spw = float((y_tr_c == 0).sum() / max(y_tr_c.sum(), 1))
    clf_meta, clf_rows, clf_metrics, roc, last_probs, all_probs = {}, [], {}, {}, {}, {}
    for key in classifier_keys():
        name = config.CLF_NAMES[key]
        log.info("Failure classifier: %s", name)
        t = time.time()
        try:
            model, params = search(build_classifier(key, spw), classifier_space(key), X_tr, y_tr_c, g_tr, args,
                                   "average_precision")
        except Exception as exc:
            log.error("   %s failed: %s", name, exc)
            continue
        secs = time.time() - t
        s_va, s_te = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
        thr = best_f1_threshold(y_va_c, s_va)
        m = {"validation": classification_metrics(y_va_c, s_va, thr),
             "test_last_cycle": classification_metrics(y_te_c[te_last], s_te[te_last], thr),
             "test_all_cycles": classification_metrics(y_te_c, s_te, thr)}
        clf_metrics[name] = m
        if len(np.unique(y_te_c[te_last])) == 2:
            roc[name] = roc_points(y_te_c[te_last], s_te[te_last])
        last_probs[key], all_probs[key] = s_te[te_last], s_te
        joblib.dump(model, config.clf_path(key))
        clf_meta[key] = {"label": name, "threshold": thr, "params": params,
                         "val_pr_auc": m["validation"]["pr_auc"], "train_seconds": round(secs, 1)}
        tm, ta = m["test_last_cycle"], m["test_all_cycles"]
        clf_rows.append({"model_key": key, "model": name,
                         **{k: tm[k] for k in ("precision", "recall", "f1", "roc_auc", "pr_auc", "threshold")},
                         **tm["confusion_matrix"], "all_cycles_f1": ta["f1"], "all_cycles_roc_auc": ta["roc_auc"],
                         "val_pr_auc": m["validation"]["pr_auc"], "train_seconds": round(secs, 1)})
        log.info("   TEST(last cycle) P %.3f | R %.3f | F1 %.3f | ROC-AUC %.3f", tm["precision"], tm["recall"],
                 tm["f1"], tm["roc_auc"])
    if not clf_meta:
        log.error("No classifier trained.")
        return 1
    best_clf = max(clf_meta, key=lambda k: clf_meta[k]["val_pr_auc"])
    log.info("Selected: RUL -> %s (val RMSE) | failure -> %s (val PR-AUC)",
             config.REG_NAMES[best_rul], config.CLF_NAMES[best_clf])

    # 5. Reports -------------------------------------------------------------------------------
    reg_df = pd.DataFrame(reg_rows).assign(selected=lambda d: d.model_key == best_rul).sort_values("rmse")
    reg_df.to_csv(config.REG_COMPARISON_PATH, index=False)
    clf_df = pd.DataFrame(clf_rows).assign(selected=lambda d: d.model_key == best_clf).sort_values("f1", ascending=False)
    clf_df.to_csv(config.CLF_COMPARISON_PATH, index=False)
    save_json({"regression": reg_metrics, "classification": clf_metrics}, config.METRICS_PATH)
    save_json(roc, config.ROC_PATH)

    thr = clf_meta[best_clf]["threshold"]
    best_reg_model = joblib.load(config.reg_path(best_rul))
    p_all = np.clip(best_reg_model.predict(X_te), 0, config.RUL_CAP)
    fleet = test[["unit", "cycle", "rul_true"]].copy()
    fleet["pred_rul"], fleet["fail_prob"] = np.round(p_all, 3), np.round(all_probs[best_clf], 5)
    fleet["status"] = [health_status(r, p, thr) for r, p in zip(fleet.pred_rul, fleet.fail_prob)]
    fleet.to_csv(config.FLEET_PATH, index=False)

    last = fleet[te_last].reset_index(drop=True)
    last["actual_at_risk"] = label_failure(last["rul_true"])
    for k, p in last_preds.items():
        last[f"pred_{k}"] = np.round(p, 3)
    for k, p in last_probs.items():
        last[f"prob_{k}"] = np.round(p, 5)
    last.to_csv(config.LAST_CYCLE_PATH, index=False)

    out = config.FIGURES_DIR
    plot_rul_scatter(y_te_last, {config.REG_NAMES[k]: v for k, v in last_preds.items()}, out)
    plot_rul_bars(reg_df.to_dict("records"), out)
    plot_confusion({n: m["test_last_cycle"] for n, m in clf_metrics.items()}, out)
    if roc:
        plot_roc(roc, out)

    # 6. Sensor importance (permutation, validation engines) ----------------------------------
    sample = rng.choice(len(X_va), min(3000, len(X_va)), replace=False)
    imp = grouped_permutation_importance(best_reg_model, X_va.iloc[sample], va["rul"].to_numpy()[sample],
                                         builder.sensors)
    imp.to_csv(config.SENSOR_IMPORTANCE_PATH, index=False)
    plot_importance(imp, out, config.REG_NAMES[best_rul])
    est = best_reg_model
    if hasattr(est, "feature_importances_"):
        pd.DataFrame({"feature": builder.feature_names, "importance": est.feature_importances_}) \
            .sort_values("importance", ascending=False).to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)
    log.info("Most important sensors: %s", ", ".join(imp.sensor.head(5)))

    save_json({
        "project": "Predictive Maintenance AI", "trained_at": datetime.now().isoformat(timespec="seconds"),
        "subset": args.subset, "best_rul_model": best_rul, "best_clf_model": best_clf,
        "regressors": reg_meta, "classifiers": clf_meta,
        "assumptions": {"rul_cap": config.RUL_CAP, "failure_window": config.FAILURE_WINDOW,
                        "warning_rul": config.WARNING_RUL, "rolling_window": config.ROLLING_WINDOW},
        "sensors_used": builder.sensors, "sensors_dropped": builder.dropped,
        "n_features": len(builder.feature_names), "missing_values": miss, "eda": eda,
        "engines": {"train": int(tr.unit.nunique()), "validation": int(va.unit.nunique()),
                    "test": int(test.unit.nunique())},
        "tuning": {"enabled": not args.no_tune, "n_iter": args.n_iter, "cv": f"GroupKFold({args.cv}) by engine"},
        "lstm_trained": bool(args.lstm and "lstm" in last_preds),
        "seed": config.RANDOM_STATE,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }, config.METADATA_PATH)
    log.info("Done in %.1f min. Dashboard: ./.venv/Scripts/python.exe -m streamlit run app.py", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
