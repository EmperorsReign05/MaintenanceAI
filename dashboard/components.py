"""Reusable widgets: gauges, prediction summary and maintenance recommendation."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.insights import recommend

from .theme import AMBER, BORDER, GREEN, MUTED, PANEL, RED, STATUS_COLORS, TEXT, note, status_badge, style_fig


def rul_gauge(rul: float) -> go.Figure:
    color = RED if rul <= config.FAILURE_WINDOW else (AMBER if rul <= config.WARNING_RUL else GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=rul,
        number={"suffix": " cycles", "valueformat": ".0f", "font": {"size": 30, "color": TEXT}},
        title={"text": "Predicted remaining useful life", "font": {"size": 13, "color": MUTED}},
        gauge={"axis": {"range": [0, config.RUL_CAP], "tickcolor": MUTED},
               "bar": {"color": color, "thickness": 0.3}, "bgcolor": PANEL, "bordercolor": BORDER,
               "steps": [{"range": [0, config.FAILURE_WINDOW], "color": "rgba(239,68,68,.15)"},
                         {"range": [config.FAILURE_WINDOW, config.WARNING_RUL], "color": "rgba(245,158,11,.12)"},
                         {"range": [config.WARNING_RUL, config.RUL_CAP], "color": "rgba(16,185,129,.10)"}]},
    ))
    return style_fig(fig, 240)


def prob_gauge(prob: float, threshold: float) -> go.Figure:
    color = RED if prob >= threshold else (AMBER if prob >= threshold / 2 else GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 30, "color": TEXT}},
        title={"text": f"Failure-risk score (fails within {config.FAILURE_WINDOW} cycles)",
               "font": {"size": 13, "color": MUTED}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED}, "bar": {"color": color, "thickness": 0.3},
               "bgcolor": PANEL, "bordercolor": BORDER,
               "threshold": {"line": {"color": TEXT, "width": 3}, "thickness": 0.85, "value": threshold * 100}},
    ))
    return style_fig(fig, 240)


def prediction_summary(pred_rul: float, fail_prob: float, status: str, threshold: float) -> None:
    c1, c2, c3 = st.columns([2, 2, 2])
    c1.plotly_chart(rul_gauge(pred_rul), width="stretch")
    c2.plotly_chart(prob_gauge(fail_prob, threshold), width="stretch")
    with c3:
        st.markdown(
            f'<div class="panel" style="margin-top:8px"><div style="color:{MUTED};font-size:.74rem;'
            f'text-transform:uppercase;letter-spacing:.08em;font-weight:600">Health status</div>'
            f'<div style="margin:12px 0">{status_badge(status)}</div>'
            f'<div style="color:{MUTED};font-size:.82rem">High risk: score &ge; {threshold:.2f} or RUL &le; '
            f'{config.FAILURE_WINDOW}<br>Warning: RUL &le; {config.WARNING_RUL} or score &ge; {threshold / 2:.2f}'
            f'</div></div>', unsafe_allow_html=True)


def recommendation(status: str, pred_rul: float, fail_prob: float, trending: list[str]) -> None:
    rec = recommend(status, pred_rul, fail_prob, trending)
    color = STATUS_COLORS[status]
    items = "".join(f"<li style='margin:4px 0'>{html.escape(a)}</li>" for a in rec["actions"])
    st.markdown(
        f'<div class="panel" style="border-left:4px solid {color}">'
        f'<div style="font-weight:700;margin-bottom:6px">Suggested maintenance actions '
        f'<span style="color:{MUTED};font-weight:500;font-size:.8rem">(model-generated)</span></div>'
        f'<ul style="margin:0 0 0 18px;padding:0">{items}</ul></div>', unsafe_allow_html=True)
    st.write("")
    note(f"<b>Not an engineering diagnosis.</b> {html.escape(rec['disclaimer'])}", warn=True)
