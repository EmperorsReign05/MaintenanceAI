"""Health status, sensor importance and model-generated maintenance suggestions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

DISCLAIMER = ("Model-generated suggestion based on simulated sensor patterns. It is NOT an engineering "
              "diagnosis and must be confirmed by qualified maintenance personnel and inspection.")

STATUS_ORDER = ["High risk", "Warning", "Healthy"]


def health_status(pred_rul: float, fail_prob: float, threshold: float) -> str:
    """High risk if the classifier flags it or predicted RUL is inside the failure window."""
    if fail_prob >= threshold or pred_rul <= config.FAILURE_WINDOW:
        return "High risk"
    if pred_rul <= config.WARNING_RUL or fail_prob >= threshold / 2:
        return "Warning"
    return "Healthy"


def recommend(status: str, pred_rul: float, fail_prob: float, trending: list[str] | None = None) -> dict:
    trending = trending or []
    names = ", ".join(f"{s} ({config.SENSOR_DESCRIPTIONS[s]})" for s in trending[:3])
    rul = max(int(round(pred_rul)), 0)
    if status == "High risk":
        actions = [
            f"Prioritise this engine for an inspection; the model estimates about {rul} cycles remaining "
            f"(failure-risk score {fail_prob:.0%}).",
            "Plan maintenance before the next operating block rather than at the routine interval.",
            "Increase monitoring frequency and review the most recent sensor readings for step changes.",
        ]
    elif status == "Warning":
        actions = [
            f"Schedule maintenance planning within the coming window (estimated RUL ~{rul} cycles).",
            "Order long-lead parts and check crew availability so maintenance can be pulled forward if needed.",
            "Monitor trends closely; re-evaluate if the risk score keeps rising.",
        ]
    else:
        actions = [
            f"No action beyond routine monitoring (estimated RUL ~{rul} cycles, capped at {config.RUL_CAP}).",
            "Keep the standard inspection schedule.",
        ]
    if names:
        actions.append(f"Sensors with the strongest recent drift: {names}.")
    return {"status": status, "actions": actions, "disclaimer": DISCLAIMER}


def trending_sensors(history: pd.DataFrame, sensors: list[str], window: int = 20) -> list[str]:
    """Sensors whose recent mean moved furthest from the engine's own early-life baseline (in std units)."""
    if len(history) < 6:
        return []
    h = history.sort_values("cycle")
    base, recent = h.head(min(window, len(h) // 2)), h.tail(min(window, len(h) // 2))
    sd = h[sensors].std().replace(0, np.nan)
    z = ((recent[sensors].mean() - base[sensors].mean()).abs() / sd).dropna()
    return z.sort_values(ascending=False).index.tolist()


def grouped_permutation_importance(model, X: pd.DataFrame, y, sensors: list[str], n_repeats: int = 3,
                                   seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Shuffle all engineered features of one sensor together and measure the RMSE increase."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    base = np.sqrt(np.mean((model.predict(X) - y) ** 2))
    groups = {s: [c for c in X.columns if c.startswith(f"{s}_")] for s in sensors}
    groups["cycle"] = ["cycle"]
    rows = []
    for name, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            deltas.append(np.sqrt(np.mean((model.predict(Xp) - y) ** 2)) - base)
        rows.append({"sensor": name, "importance": float(np.mean(deltas)), "std": float(np.std(deltas)),
                     "label": f"{name} - {config.SENSOR_DESCRIPTIONS.get(name, 'operating cycle')}"})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
