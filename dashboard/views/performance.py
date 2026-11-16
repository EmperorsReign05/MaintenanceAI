"""Model performance: RUL regression and failure classification comparisons."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import config

from .. import data
from ..theme import MUTED, SERIES, header, note, style_fig


def _rul(meta, v) -> None:
    reg, last = data.reg_comparison(v), data.last_cycle(v)
    best = reg[reg.selected].iloc[0]
    c = st.columns(4)
    c[0].metric(f"RMSE - {best.model}", f"{best.rmse:.2f}")
    c[1].metric("MAE", f"{best.mae:.2f}")
    c[2].metric("R²", f"{best.r2:.3f}")
    c[3].metric("NASA score", f"{best.nasa_score:,.0f}", help="PHM08 asymmetric score; late predictions are "
                                                             "penalised more than early ones. Lower is better.")
    st.dataframe(reg[["model", "rmse", "mae", "r2", "nasa_score", "val_rmse", "test_all_rmse", "train_seconds",
                      "deployable", "selected"]], hide_index=True, width="stretch", column_config={
        "model": "Model", "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
        "mae": st.column_config.NumberColumn("MAE", format="%.2f"), "r2": st.column_config.NumberColumn("R²", format="%.3f"),
        "nasa_score": st.column_config.NumberColumn("NASA score", format="%.0f"),
        "val_rmse": st.column_config.NumberColumn("Val. RMSE", format="%.2f"),
        "test_all_rmse": st.column_config.NumberColumn("RMSE all cycles", format="%.2f"),
        "train_seconds": st.column_config.NumberColumn("Train s", format="%.1f"),
        "deployable": st.column_config.CheckboxColumn("Used in app"),
        "selected": st.column_config.CheckboxColumn("Selected")})
    st.caption("Main metrics: last observed cycle of each test engine vs NASA's true RUL (standard C-MAPSS "
               "protocol). 'All cycles' scores every test cycle against the capped RUL label.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for metric, color in zip(["rmse", "mae"], SERIES):
            fig.add_bar(name=metric.upper(), x=reg.model, y=reg[metric], marker_color=color,
                        text=[f"{x:.1f}" for x in reg[metric]], textposition="outside")
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Cycles (lower is better)")
        st.plotly_chart(style_fig(fig, 360, "RUL error by model"), width="stretch")
    with c2:
        choice = st.selectbox("Predicted vs actual for", reg.model_key.tolist(),
                              format_func=lambda k: config.REG_NAMES[k])
        col = f"pred_{choice}"
        if col in last:
            fig = go.Figure(go.Scatter(x=last.rul_true, y=last[col], mode="markers",
                                       marker=dict(color=SERIES[0], size=8, opacity=0.8),
                                       hovertemplate="Actual %{x}<br>Pred %{y:.0f}<extra></extra>"))
            hi = max(last.rul_true.max(), last[col].max()) + 5
            fig.add_scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=MUTED, dash="dash"), showlegend=False)
            fig.update_xaxes(title="Actual RUL")
            fig.update_yaxes(title="Predicted RUL")
            st.plotly_chart(style_fig(fig, 310), width="stretch")


def _clf(meta, v) -> None:
    clf, roc = data.clf_comparison(v), data.json_report(str(config.ROC_PATH), v)
    st.dataframe(clf[["model", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "tp", "fp", "fn", "tn",
                      "all_cycles_f1", "all_cycles_roc_auc", "selected"]], hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in
                                ["precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "all_cycles_f1",
                                 "all_cycles_roc_auc"]} | {"selected": st.column_config.CheckboxColumn("Selected")})
    st.caption(f"Positive class = engine fails within {config.FAILURE_WINDOW} cycles. Thresholds maximise F1 on "
               "validation engines. With few test engines (one row each), last-cycle scores move in large steps; "
               "the all-cycles columns use every test cycle.")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for metric, color in zip(["precision", "recall", "f1", "roc_auc"], SERIES):
            fig.add_bar(name=metric.replace("_", "-").upper(), x=clf.model, y=clf[metric], marker_color=color)
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(style_fig(fig, 360, "Classifier comparison"), width="stretch")
    with c2:
        if roc:
            fig = go.Figure()
            for (name, c), color in zip(roc.items(), SERIES):
                fig.add_scatter(x=c["fpr"], y=c["tpr"], name=name, line=dict(color=color, width=2))
            fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash="dash"), name="Chance")
            fig.update_xaxes(title="False positive rate")
            fig.update_yaxes(title="True positive rate")
            st.plotly_chart(style_fig(fig, 360, "ROC curve"), width="stretch")

    name = st.selectbox("Confusion matrix for", clf.model.tolist(), index=int(np.argmax(clf.selected.values)))
    r = clf[clf.model == name].iloc[0]
    mat = np.array([[r.tn, r.fp], [r.fn, r.tp]])
    fig = go.Figure(go.Heatmap(z=mat, x=["Pred. healthy", "Pred. at risk"], y=["Actual healthy", "Actual at risk"],
                               text=mat, texttemplate="%{text}", textfont=dict(size=20), colorscale="Blues",
                               showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    c1, c2 = st.columns([2, 3])
    c1.plotly_chart(style_fig(fig, 300), width="stretch")
    c2.markdown(f"""
- **{r.tp}** at-risk engines correctly flagged
- **{r.fn}** at-risk engines missed (most costly in maintenance)
- **{r.fp}** healthy engines flagged early (cost: unneeded inspection)
- Decision threshold **{r.threshold:.3f}**
""")


def render() -> None:
    v = data.version()
    meta = data.metadata(v)
    header("Model Performance", "All figures computed from test-set predictions produced by train.py")
    t1, t2 = st.tabs(["Remaining useful life (regression)", "Failure risk (classification)"])
    with t1:
        _rul(meta, v)
    with t2:
        _clf(meta, v)
    note("Models were tuned with engine-grouped cross-validation and selected on validation engines "
         "(RUL by RMSE, failure risk by PR-AUC), so the test engines remain an unbiased check.")
