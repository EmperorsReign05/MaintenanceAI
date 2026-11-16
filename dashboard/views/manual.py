"""Manual prediction: enter a sensor reading (optionally on top of an engine's history) and score it
with exactly the same preprocessing pipeline used in training."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.insights import trending_sensors

from .. import data
from ..components import prediction_summary, recommendation
from ..theme import header, note

NO_HISTORY = "No history (single reading)"


def _score(history: pd.DataFrame, meta: dict, v: float) -> None:
    try:
        res = data.predictor(v).predict_latest(history)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return
    thr = meta["classifiers"][meta["best_clf_model"]]["threshold"]
    prediction_summary(res["pred_rul"], res["fail_prob"], res["status"], thr)
    trending = trending_sensors(history, meta["sensors_used"]) if len(history) > 5 else []
    recommendation(res["status"], res["pred_rul"], res["fail_prob"], trending)


def _form(meta: dict, v: float) -> None:
    _, test = data.raw_data(meta["subset"], v)
    pred = data.predictor(v)
    options = [NO_HISTORY] + ([f"Engine {u}" for u in sorted(test.unit.unique())] if test is not None else [])
    base = st.selectbox("Start from", options, index=1 if len(options) > 1 else 0,
                        help="Rolling-trend features need past cycles. Choosing an engine appends your reading "
                             "to its real history.")
    if base == NO_HISTORY:
        history = None
        ranges = pred.builder.sensor_ranges
        last = {c: (lo + hi) / 2 for c, (lo, hi) in ranges.items()}
        last["cycle"] = 1
        note("Without history, rolling mean/std/slope features are computed from this single reading, so "
             "trend information is missing and predictions are less reliable.", warn=True)
    else:
        unit = int(base.split()[-1])
        history = test[test.unit == unit][config.COLUMNS].sort_values("cycle")
        last = history.iloc[-1].to_dict()
        st.caption(f"Engine {unit} has {len(history)} recorded cycles. Fields are pre-filled with its latest reading.")

    with st.form("reading"):
        mode = "append"
        if history is not None:
            mode = st.radio("Use this reading to", ["append", "replace"], horizontal=True,
                            format_func=lambda m: "Append as the next cycle" if m == "append" else "Replace the latest cycle")
        else:
            st.number_input("Cycle number", min_value=1, value=1, step=1, key="m_cycle")
        st.markdown("**Operating settings**")
        cols = st.columns(3)
        vals = {}
        for col, c in zip(cols, config.OP_SETTINGS):
            vals[c] = col.number_input(c, value=float(last[c]), format="%.4f")
        st.markdown("**Sensors** (sensors constant in training are fixed to their training value)")
        cols = st.columns(4)
        for i, s in enumerate(config.SENSORS):
            disabled = s in meta["sensors_dropped"]
            vals[s] = cols[i % 4].number_input(f"{s} - {config.SENSOR_DESCRIPTIONS[s]}", value=float(last[s]),
                                               format="%.4f", disabled=disabled)
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")
    if not submitted:
        return

    bad = pred.builder.out_of_range(vals)
    if bad:
        note("Outside the range seen in training (the model is extrapolating, treat the result with caution): "
             + ", ".join(bad), warn=True)

    if history is None:
        row = {"unit": 0, "cycle": int(st.session_state.get("m_cycle", 1)), **vals}
        hist = pd.DataFrame([row])[config.COLUMNS]
    else:
        hist = history.copy()
        nxt = int(hist.cycle.max()) + 1 if mode == "append" else int(hist.cycle.max())
        if mode == "replace":
            hist = hist[hist.cycle != nxt]
        new = {"unit": int(hist.unit.iloc[0]) if len(hist) else 0, "cycle": nxt, **vals}
        hist = pd.concat([hist, pd.DataFrame([new])], ignore_index=True)[config.COLUMNS]
    _score(hist, meta, v)


def _upload(meta: dict, v: float) -> None:
    st.markdown("Upload one engine's history as CSV with columns: "
                f"`{', '.join(config.COLUMNS[:5])}, s1 ... s21`. Raw C-MAPSS .txt files are also accepted.")
    f = st.file_uploader("Engine history", type=["csv", "txt"])
    if f is None:
        return
    try:
        if f.name.endswith(".txt"):
            df = pd.read_csv(f, sep=r"\s+", header=None).dropna(axis=1, how="all")
            df.columns = config.COLUMNS
        else:
            df = pd.read_csv(f)
        units = df["unit"].unique() if "unit" in df.columns else [0]
        if len(units) > 1:
            u = st.selectbox("File contains several engines - choose one", sorted(units))
            df = df[df.unit == u]
        st.caption(f"{len(df)} cycles loaded.")
        _score(df, meta, v)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")


def render() -> None:
    v = data.version()
    meta = data.metadata(v)
    header("Manual Prediction", "Score a new reading with the same preprocessing pipeline used in training")
    t1, t2 = st.tabs(["Enter a reading", "Upload engine history"])
    with t1:
        _form(meta, v)
    with t2:
        _upload(meta, v)
