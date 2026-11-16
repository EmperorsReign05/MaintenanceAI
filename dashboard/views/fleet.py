"""Fleet overview: health of every monitored engine at its latest cycle."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.insights import STATUS_ORDER

from .. import data
from ..theme import AMBER, BLUE, GREEN, MUTED, RED, STATUS_COLORS, header, kpi, note, style_fig


def render() -> None:
    v = data.version()
    meta, last = data.metadata(v), data.last_cycle(v)
    header("Fleet Overview", f"{meta['subset']} test fleet - each engine assessed at its most recent cycle "
                             f"(RUL: {config.REG_NAMES[meta['best_rul_model']]}, risk: "
                             f"{config.CLF_NAMES[meta['best_clf_model']]})")

    counts = last.status.value_counts()
    n = len(last)
    c = st.columns(5)
    with c[0]:
        kpi("Engines monitored", f"{n}", "test trajectories", BLUE)
    with c[1]:
        kpi("Healthy", f"{counts.get('Healthy', 0)}", f"{100 * counts.get('Healthy', 0) / n:.0f}% of fleet", GREEN)
    with c[2]:
        kpi("Warning", f"{counts.get('Warning', 0)}", f"RUL <= {config.WARNING_RUL} cycles", AMBER)
    with c[3]:
        kpi("High risk", f"{counts.get('High risk', 0)}", f"failure within ~{config.FAILURE_WINDOW} cycles", RED)
    with c[4]:
        kpi("Median predicted RUL", f"{last.pred_rul.median():.0f}", f"cycles (capped at {config.RUL_CAP})", "#A78BFA")
    st.write("")

    c1, c2 = st.columns([3, 2])
    with c1:
        srt = last.sort_values("pred_rul")
        fig = go.Figure(go.Bar(
            x=srt.unit.astype(str), y=srt.pred_rul, marker_color=[STATUS_COLORS[s] for s in srt.status],
            customdata=np.stack([srt.fail_prob, srt.status], axis=1),
            hovertemplate="Engine %{x}<br>Pred. RUL %{y:.0f}<br>Risk %{customdata[0]:.1%}<br>%{customdata[1]}"
                          "<extra></extra>"))
        fig.add_hline(y=config.FAILURE_WINDOW, line_dash="dash", line_color=RED)
        fig.add_hline(y=config.WARNING_RUL, line_dash="dot", line_color=AMBER)
        fig.update_xaxes(title="Engine (sorted by predicted RUL)", type="category", showticklabels=n <= 60)
        fig.update_yaxes(title="Predicted RUL (cycles)")
        st.plotly_chart(style_fig(fig, 380, "Predicted RUL per engine"), width="stretch")
    with c2:
        vals = [counts.get(s, 0) for s in STATUS_ORDER]
        fig = go.Figure(go.Pie(labels=STATUS_ORDER, values=vals, hole=0.62, sort=False,
                               marker_colors=[STATUS_COLORS[s] for s in STATUS_ORDER], textinfo="label+value"))
        st.plotly_chart(style_fig(fig, 380, "Fleet health status"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(last, x="rul_true", y="pred_rul", color="status", color_discrete_map=STATUS_COLORS,
                         hover_data=["unit", "fail_prob"], labels={"rul_true": "Actual RUL (NASA ground truth)",
                                                                   "pred_rul": "Predicted RUL"})
        hi = max(last.rul_true.max(), config.RUL_CAP) + 5
        fig.add_scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(dash="dash", color=MUTED), name="Perfect")
        st.plotly_chart(style_fig(fig, 360, "Predicted vs actual RUL"), width="stretch")
    with c2:
        fig = px.histogram(last, x="fail_prob", nbins=25, color_discrete_sequence=[BLUE],
                           labels={"fail_prob": "Failure-risk score"})
        thr = meta["classifiers"][meta["best_clf_model"]]["threshold"]
        fig.add_vline(x=thr, line_dash="dash", line_color=RED, annotation_text=f"threshold {thr:.2f}")
        st.plotly_chart(style_fig(fig, 360, "Failure-risk score distribution"), width="stretch")

    st.subheader("Fleet register")
    order = {s: i for i, s in enumerate(STATUS_ORDER)}
    table = last.assign(_o=last.status.map(order)).sort_values(["_o", "pred_rul"])[
        ["unit", "cycle", "status", "pred_rul", "fail_prob", "rul_true"]]
    st.dataframe(table, hide_index=True, width="stretch", height=360, column_config={
        "unit": "Engine", "cycle": "Cycles observed", "status": "Status",
        "pred_rul": st.column_config.NumberColumn("Predicted RUL", format="%.0f"),
        "fail_prob": st.column_config.ProgressColumn("Failure risk", min_value=0.0, max_value=1.0, format="%.2f"),
        "rul_true": st.column_config.NumberColumn("Actual RUL", format="%d"),
    })
    note("Actual RUL comes from NASA's RUL file for the test set and is shown only to judge the model; "
         "in live operation it would be unknown.")
