"""Sensor analysis: degradation patterns, healthy vs degrading engines, importance."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import config

from .. import data
from ..theme import BLUE, CYAN, GREEN, MUTED, RED, header, note, style_fig


def render() -> None:
    v = data.version()
    meta = data.metadata(v)
    header("Sensor Analysis", "How sensor readings evolve as engines approach failure (training engines)")
    train, _ = data.raw_data(meta["subset"], v)
    if train is None:
        st.warning("Raw C-MAPSS files not found in data/. Place them there to enable this page.")
        return

    sensors = meta["sensors_used"]
    s = st.selectbox("Sensor", sensors, format_func=lambda x: f"{x} - {config.SENSOR_DESCRIPTIONS[x]}")

    c1, c2 = st.columns([3, 2])
    with c1:
        horizon = int(min(train.groupby("unit").cycle.max().quantile(0.5), 250))
        t = train[train.rul_true <= horizon]
        g = t.groupby("rul_true")[s]
        mean, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
        x = -mean.index.to_numpy()
        fig = go.Figure()
        fig.add_scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([hi.values, lo.values[::-1]]),
                        fill="toself", fillcolor="rgba(34,211,238,.15)", line=dict(width=0), name="10-90% band")
        fig.add_scatter(x=x, y=mean.values, line=dict(color=CYAN, width=2.5), name="Fleet mean")
        rng = np.random.default_rng(0)
        for u in rng.choice(train.unit.unique(), 3, replace=False):
            e = train[(train.unit == u) & (train.rul_true <= horizon)]
            fig.add_scatter(x=-e.rul_true, y=e[s], line=dict(width=1), opacity=0.55, name=f"Engine {u}")
        fig.add_vrect(x0=-config.FAILURE_WINDOW, x1=0, fillcolor=RED, opacity=0.08, line_width=0)
        fig.update_xaxes(title="Cycles before failure")
        st.plotly_chart(style_fig(fig, 400, f"Degradation pattern - {s}"), width="stretch")
    with c2:
        healthy = train.loc[train.rul_true > 100, s]
        degr = train.loc[train.rul_true <= config.FAILURE_WINDOW, s]
        fig = go.Figure()
        fig.add_histogram(x=healthy, name="Healthy (RUL > 100)", marker_color=GREEN, opacity=0.65,
                          histnorm="probability density")
        fig.add_histogram(x=degr, name=f"Degrading (RUL <= {config.FAILURE_WINDOW})", marker_color=RED,
                          opacity=0.65, histnorm="probability density")
        fig.update_layout(barmode="overlay")
        st.plotly_chart(style_fig(fig, 400, "Healthy vs degrading engines"), width="stretch")
        pooled = np.sqrt((healthy.var() + degr.var()) / 2)
        if pooled > 0:
            st.caption(f"Mean shift: {degr.mean() - healthy.mean():+.3f} ({(degr.mean() - healthy.mean()) / pooled:+.2f} "
                       "pooled standard deviations).")

    c1, c2 = st.columns(2)
    with c1:
        corr = train[sensors + ["rul_true"]].corr()["rul_true"].drop("rul_true").sort_values()
        fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                               marker_color=[RED if x < 0 else BLUE for x in corr.values]))
        fig.update_xaxes(title="Pearson correlation with RUL")
        st.plotly_chart(style_fig(fig, 420, "Sensor correlation with remaining life"), width="stretch")
    with c2:
        imp = data.sensor_importance(v)
        if imp is not None:
            top = imp.head(12).iloc[::-1]
            fig = go.Figure(go.Bar(x=top.importance, y=top.label, orientation="h", marker_color=CYAN,
                                   error_x=dict(type="data", array=top["std"], color=MUTED)))
            fig.update_xaxes(title="RMSE increase when shuffled (cycles)")
            st.plotly_chart(style_fig(fig, 420, f"Feature importance - {config.REG_NAMES[meta['best_rul_model']]}"),
                            width="stretch")
    note("<b>Feature importance</b> is permutation importance: all engineered features of one sensor (raw, rolling "
         "mean, std, slope) are shuffled together on validation engines, and the increase in RUL error is measured. "
         f"Sensors dropped as constant in training: {', '.join(meta['sensors_dropped']) or 'none'}.")
