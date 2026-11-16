"""Metrics and comparison plots. All numbers are computed from real predictions."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)

from .eda import BLUE, CYAN, RED, set_style

PALETTE = [BLUE, "#F59E0B", "#10B981", RED, "#A78BFA"]


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score: late predictions (pred > true) are penalised more than early ones."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": int(len(y_true)),
    }


def best_f1_threshold(y_true, scores) -> float:
    p, r, t = precision_recall_curve(y_true, scores)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[int(np.nanargmax(f1[:-1]))])


def classification_metrics(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, scores)) if both else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)), "n_positive": int(y_true.sum()),
    }


def roc_points(y_true, scores) -> dict:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {"fpr": np.round(fpr, 5).tolist(), "tpr": np.round(tpr, 5).tolist()}


def plot_rul_scatter(y_true, preds: dict, out: Path) -> None:
    set_style()
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    hi = max(float(np.max(y_true)), max(float(np.max(p)) for p in preds.values())) + 5
    for ax, (name, p), c in zip(axes[0], preds.items(), PALETTE):
        ax.scatter(y_true, p, s=14, alpha=0.75, color=c)
        ax.plot([0, hi], [0, hi], "--", color="#6B7280")
        rmse = np.sqrt(mean_squared_error(y_true, p))
        ax.set(title=f"{name}\nRMSE {rmse:.2f}", xlabel="True RUL", ylabel="Predicted RUL", xlim=(0, hi), ylim=(0, hi))
    fig.tight_layout()
    fig.savefig(out / "07_rul_pred_vs_true.png", dpi=130)
    plt.close(fig)


def plot_rul_bars(rows: list[dict], out: Path) -> None:
    set_style()
    names = [r["model"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, label in zip(axes, ["rmse", "mae", "r2"], ["RMSE (lower is better)", "MAE (lower is better)",
                                                           "R² (higher is better)"]):
        vals = [r[key] for r in rows]
        bars = ax.bar(names, vals, color=PALETTE[: len(names)])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "08_rul_model_comparison.png", dpi=130)
    plt.close(fig)


def plot_confusion(metrics: dict, out: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4), squeeze=False)
    for ax, (name, m) in zip(axes[0], metrics.items()):
        cm = m["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if v > mat.max() * 0.55 else "#111827")
        ax.set_xticks([0, 1], ["Healthy", "At risk"])
        ax.set_yticks([0, 1], ["Healthy", "At risk"])
        ax.set(xlabel="Predicted", ylabel="Actual", title=f"{name}\nthreshold {m['threshold']:.2f}")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out / "09_failure_confusion_matrices.png", dpi=130)
    plt.close(fig)


def plot_roc(curves: dict, out: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, c), col in zip(curves.items(), PALETTE):
        ax.plot(c["fpr"], c["tpr"], lw=2, color=col, label=name)
    ax.plot([0, 1], [0, 1], "--", color="#6B7280")
    ax.set(title="Failure-risk ROC (test engines, last cycle)", xlabel="False positive rate",
           ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "10_failure_roc.png", dpi=130)
    plt.close(fig)


def plot_importance(sensor_imp, out: Path, model_label: str) -> None:
    set_style()
    top = sensor_imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["label"], top["importance"], color=CYAN)
    ax.set_title(f"Sensor importance - {model_label}\n(permutation importance, validation engines)")
    ax.set_xlabel("Increase in RMSE when the sensor's features are shuffled (cycles)")
    fig.tight_layout()
    fig.savefig(out / "11_sensor_importance.png", dpi=130)
    plt.close(fig)
