"""Industrial charcoal/blue theme: colours, CSS, Plotly styling, HTML building blocks."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

CHARCOAL = "#15171C"
PANEL = "#1E2128"
BORDER = "#2C313A"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
BLUE = "#3B82F6"
CYAN = "#22D3EE"
GREEN = "#10B981"
AMBER = "#F59E0B"
RED = "#EF4444"
STATUS_COLORS = {"Healthy": GREEN, "Warning": AMBER, "High risk": RED}
SERIES = [BLUE, AMBER, GREEN, RED, "#A78BFA", CYAN]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.8rem; max-width: 1450px; }}
section[data-testid="stSidebar"] {{ background: #111317; border-right: 1px solid {BORDER}; }}
.pm-header {{ border-left: 4px solid {BLUE}; padding-left: 14px; margin-bottom: 1.2rem; }}
.pm-header h1 {{ font-size: 1.6rem; font-weight: 700; margin: 0; color: {TEXT}; }}
.pm-header p {{ color: {MUTED}; margin: .2rem 0 0 0; font-size: .92rem; }}
.kpi {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px;
        border-left: 4px solid var(--accent); height: 100%; }}
.kpi .l {{ color: {MUTED}; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }}
.kpi .v {{ color: {TEXT}; font-size: 1.9rem; font-weight: 700; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }}
.kpi .s {{ color: {MUTED}; font-size: .78rem; margin-top: 4px; }}
.sensor {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 14px; }}
.sensor .n {{ color: {MUTED}; font-size: .74rem; font-weight: 600; }}
.sensor .v {{ color: {TEXT}; font-size: 1.25rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
.sensor .d {{ font-size: .78rem; font-weight: 600; }}
.panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px; }}
.status {{ display: inline-flex; align-items: center; gap: 8px; padding: 5px 14px; border-radius: 6px;
           font-weight: 700; font-size: .9rem; letter-spacing: .02em; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
.note {{ background: rgba(59,130,246,.08); border: 1px solid rgba(59,130,246,.35); border-radius: 8px;
         padding: 11px 15px; font-size: .86rem; color: {TEXT}; }}
.warnbox {{ background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.45); border-radius: 8px;
            padding: 11px 15px; font-size: .86rem; color: {TEXT}; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 10px 14px; }}
</style>
"""


def apply_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="pm-header"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", accent: str = BLUE) -> None:
    st.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="l">{html.escape(label)}</div>'
                f'<div class="v">{html.escape(value)}</div><div class="s">{html.escape(sub)}</div></div>',
                unsafe_allow_html=True)


def status_badge(status: str) -> str:
    c = STATUS_COLORS.get(status, MUTED)
    return (f'<span class="status" style="background:{c}1F;color:{c};border:1px solid {c}66">'
            f'<span class="dot" style="background:{c}"></span>{html.escape(status.upper())}</span>')


def sensor_card(name: str, desc: str, value: float, delta_pct: float | None) -> None:
    if delta_pct is None:
        d = f'<span class="d" style="color:{MUTED}">no baseline</span>'
    else:
        c = RED if abs(delta_pct) >= 1.0 else (AMBER if abs(delta_pct) >= 0.3 else GREEN)
        d = f'<span class="d" style="color:{c}">{delta_pct:+.2f}% vs early life</span>'
    st.markdown(f'<div class="sensor"><div class="n">{html.escape(name)} &middot; {html.escape(desc)}</div>'
                f'<div class="v">{value:,.2f}</div>{d}</div>', unsafe_allow_html=True)


def note(text_html: str, warn: bool = False) -> None:
    st.markdown(f'<div class="{"warnbox" if warn else "note"}">{text_html}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(size=14, color=TEXT)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL, font=dict(family="Inter, sans-serif", color=TEXT),
        margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h", y=-0.18),
        hoverlabel=dict(bgcolor=CHARCOAL, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED)
    return fig
