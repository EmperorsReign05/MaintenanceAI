"""Engine inspector: sensor history, degradation trend and predicted health for one engine."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.insights import trending_sensors

from .. import data
from ..components import prediction_summary, recommendation
from ..theme import AMBER, CYAN, MUTED, RED, SERIES, header, sensor_card, style_fig


def render() -> None:
    v = data.version()
    meta, fleet, last = data.metadata(v), data.fleet(v), data.last_cycle(v)
    header("Engine Inspector", "Select an engine to inspect its sensor history and predicted health")

    order = last.sort_values(["fail_prob"], ascending=False)
    labels = {int(r.unit): f"Engine {int(r.unit)}  -  {r.status}  -  RUL {r.pred_rul:.0f}" for r in order.itertuples()}
    unit = st.selectbox("Engine (sorted by failure risk)", list(labels), format_func=labels.get)
    hist = fleet[fleet.unit == unit].sort_values("cycle")
    cur = hist.iloc[-1]
    thr = meta["classifiers"][meta["best_clf_model"]]["threshold"]

    _, test = data.raw_data(meta["subset"], v)
    raw = test[test.unit == unit].sort_values("cycle") if test is not None else None
    trending = trending_sensors(raw, meta["sensors_used"]) if raw is not None else []

    prediction_summary(cur.pred_rul, cur.fail_prob, cur.status, thr)
    st.caption(f"Engine {unit} - {int(cur.cycle)} cycles observed - actual RUL (NASA ground truth) "
               f"{int(cur.rul_true)} cycles")

    if raw is not None:
        st.markdown("##### Latest sensor readings")
        show = (trending or meta["sensors_used"])[:6]
        cols = st.columns(len(show))
        base = raw.head(min(20, max(len(raw) // 3, 1)))
        for col, s in zip(cols, show):
            b = base[s].mean()
            delta = 100 * (raw[s].iloc[-1] - b) / abs(b) if abs(b) > 1e-9 else None
            with col:
                sensor_card(s, config.SENSOR_DESCRIPTIONS[s], float(raw[s].iloc[-1]), delta)
        st.caption("Cards show the sensors that drifted most from this engine's early-life baseline.")
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=hist.cycle, y=hist.rul_true.clip(upper=config.RUL_CAP), name="Actual RUL (capped)",
                        line=dict(color=MUTED, dash="dash"))
        fig.add_scatter(x=hist.cycle, y=hist.pred_rul, name="Predicted RUL", line=dict(color=CYAN, width=2.5))
        fig.add_hrect(y0=0, y1=config.FAILURE_WINDOW, fillcolor=RED, opacity=0.08, line_width=0)
        fig.add_hrect(y0=config.FAILURE_WINDOW, y1=config.WARNING_RUL, fillcolor=AMBER, opacity=0.06, line_width=0)
        fig.update_xaxes(title="Cycle")
        fig.update_yaxes(title="RUL (cycles)")
        st.plotly_chart(style_fig(fig, 340, "Degradation trend - remaining useful life"), width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=hist.cycle, y=hist.fail_prob, name="Failure-risk score", fill="tozeroy",
                        line=dict(color=RED, width=2))
        fig.add_hline(y=thr, line_dash="dash", line_color="#E5E7EB", annotation_text=f"threshold {thr:.2f}")
        fig.update_xaxes(title="Cycle")
        fig.update_yaxes(title="Score", range=[0, 1.02])
        st.plotly_chart(style_fig(fig, 340, "Failure-risk score over time"), width="stretch")

    if raw is not None:
        st.markdown("##### Sensor monitoring")
        default = (trending or meta["sensors_used"])[:3]
        chosen = st.multiselect("Sensors", meta["sensors_used"], default=default,
                                format_func=lambda s: f"{s} - {config.SENSOR_DESCRIPTIONS[s]}")
        smooth = st.toggle("Smooth (5-cycle rolling mean)", value=True)
        for i in range(0, len(chosen), 2):
            cols = st.columns(2)
            for col, s, color in zip(cols, chosen[i:i + 2], SERIES[i % len(SERIES):] + SERIES):
                y = raw[s].rolling(5, min_periods=1).mean() if smooth else raw[s]
                fig = go.Figure(go.Scatter(x=raw.cycle, y=y, line=dict(color=color, width=2), name=s))
                fig.update_xaxes(title="Cycle")
                col.plotly_chart(style_fig(fig, 260, f"{s} - {config.SENSOR_DESCRIPTIONS[s]}"), width="stretch")
    else:
        st.info("Raw sensor files not found in data/ - sensor history charts are hidden.")

    st.markdown("##### Maintenance recommendation")
    recommendation(cur.status, cur.pred_rul, cur.fail_prob, trending)
