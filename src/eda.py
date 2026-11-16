"""Exploratory analysis of C-MAPSS training engines; figures saved to reports/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .utils import get_logger, save_json

log = get_logger()

CHARCOAL = "#15171C"
PANEL = "#1E2128"
GRID = "#2C313A"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
BLUE = "#3B82F6"
CYAN = "#22D3EE"
RED = "#EF4444"
AMBER = "#F59E0B"
GREEN = "#10B981"


def set_style() -> None:
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        "figure.facecolor": CHARCOAL, "savefig.facecolor": CHARCOAL, "axes.facecolor": PANEL,
        "axes.edgecolor": GRID, "grid.color": GRID, "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 12,
        "legend.facecolor": PANEL, "legend.edgecolor": GRID, "font.size": 10,
    })


def _save(fig, name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)


def sensor_rul_correlation(train: pd.DataFrame, sensors: list[str]) -> pd.Series:
    return train[sensors + ["rul_true"]].corr()["rul_true"].drop("rul_true").sort_values()


def run_eda(train: pd.DataFrame, informative: list[str], subset: str, out: Path = config.FIGURES_DIR) -> dict:
    set_style()
    life = train.groupby("unit")["cycle"].max()
    std = train[config.SENSORS].std()
    corr = sensor_rul_correlation(train, informative)
    top = corr.abs().sort_values(ascending=False).index[:6].tolist()

    summary = {
        "subset": subset, "n_engines": int(train.unit.nunique()), "n_rows": int(len(train)),
        "missing_values": int(train.isna().sum().sum()),
        "lifetime_min": int(life.min()), "lifetime_mean": float(life.mean()), "lifetime_max": int(life.max()),
        "informative_sensors": informative,
        "constant_sensors": [s for s in config.SENSORS if s not in informative],
        "top_sensors_by_rul_correlation": {s: round(float(corr[s]), 4) for s in top},
    }

    # 1 lifetimes
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(life, bins=25, color=BLUE, edgecolor=CHARCOAL)
    ax.set(title=f"Engine lifetimes ({subset} training set)", xlabel="Cycles until failure", ylabel="Engines")
    _save(fig, "01_engine_lifetimes.png", out)

    # 2 sensor variability
    fig, ax = plt.subplots(figsize=(8, 4))
    cv = (std / train[config.SENSORS].mean().abs().replace(0, 1)).fillna(0)
    ax.bar(config.SENSORS, cv, color=[CYAN if s in informative else "#4B5563" for s in config.SENSORS])
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set(title="Sensor variability (grey = dropped as non-informative)", ylabel="Coefficient of variation")
    ax.tick_params(axis="x", rotation=60)
    _save(fig, "02_sensor_variability.png", out)

    # 3 degradation trajectories vs cycles-to-failure
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    rng = np.random.default_rng(config.RANDOM_STATE)
    units = rng.choice(train.unit.unique(), size=min(8, train.unit.nunique()), replace=False)
    for ax, s in zip(axes.ravel(), top):
        for u in units:
            e = train[train.unit == u]
            ax.plot(-e["rul_true"], e[s].rolling(5, min_periods=1).mean(), lw=1, alpha=0.8)
        ax.set(title=f"{s} - {config.SENSOR_DESCRIPTIONS[s]}", xlabel="Cycles before failure")
        ax.title.set_fontsize(9)
    fig.suptitle("Sensor degradation trajectories (8 random engines, 5-cycle smoothing)", color=TEXT)
    _save(fig, "03_degradation_trajectories.png", out)

    # 4 correlation with RUL
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(corr.index, corr.values, color=[RED if v < 0 else BLUE for v in corr.values])
    ax.set(title="Correlation of each sensor with true RUL", xlabel="Pearson r")
    _save(fig, "04_sensor_rul_correlation.png", out)

    # 5 healthy vs degrading
    phase = np.where(train["rul_true"] > 100, "Healthy (RUL > 100)",
                     np.where(train["rul_true"] <= config.FAILURE_WINDOW, f"Degrading (RUL <= {config.FAILURE_WINDOW})", None))
    sub = train.assign(phase=phase).dropna(subset=["phase"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, s in zip(axes.ravel(), top):
        sns.kdeplot(data=sub, x=s, hue="phase", ax=ax, fill=True, common_norm=False, alpha=0.4,
                    palette={"Healthy (RUL > 100)": GREEN, f"Degrading (RUL <= {config.FAILURE_WINDOW})": RED},
                    warn_singular=False)
        ax.set_title(s)
    fig.suptitle("Healthy vs degrading engines: sensor distributions", color=TEXT)
    _save(fig, "05_healthy_vs_degrading.png", out)

    # 6 operating settings
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(train["op1"], train["op2"], s=3, alpha=0.4, color=BLUE)
    ax.set(title="Operating settings (clusters = operating conditions)", xlabel="op1 (altitude-related)",
           ylabel="op2 (Mach-related)")
    _save(fig, "06_operating_conditions.png", out)

    save_json(summary, config.EDA_SUMMARY_PATH)
    log.info("EDA figures saved to %s", out)
    return summary
