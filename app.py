"""Predictive Maintenance AI dashboard. Run: ./.venv/Scripts/python.exe -m streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Predictive Maintenance AI", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="expanded")

from dashboard import data  # noqa: E402
from dashboard.theme import MUTED, apply_css  # noqa: E402
from dashboard.views import about, engine, fleet, manual, performance, sensors  # noqa: E402
from src import config  # noqa: E402

PAGES = {
    "Fleet Overview": fleet.render,
    "Engine Inspector": engine.render,
    "Sensor Analysis": sensors.render,
    "Manual Prediction": manual.render,
    "Model Performance": performance.render,
    "Data & Methodology": about.render,
}


def setup_screen() -> None:
    st.title("Predictive Maintenance AI")
    st.error("No trained models found yet, so there is nothing real to display.")
    st.markdown(f"""
1. Download **CMAPSSData.zip** from [NASA]({config.DATASET_URL}) and extract `train_FD001.txt`,
   `test_FD001.txt` and `RUL_FD001.txt` into `data/`.
2. Train:
```powershell
./.venv/Scripts/python.exe train.py --no-tune   # quick first run
./.venv/Scripts/python.exe train.py             # full tuned run
```
3. Refresh this page.
""")


def main() -> None:
    apply_css()
    with st.sidebar:
        st.markdown("## ⚙️ Predictive Maintenance AI")
        st.markdown(f"<span style='color:{MUTED};font-size:.85rem'>Turbofan failure prediction & RUL "
                    "estimation</span>", unsafe_allow_html=True)
        st.write("")
    if not data.ready():
        setup_screen()
        return
    try:
        meta = data.metadata(data.version())
    except Exception as exc:
        st.error(f"Could not read metadata ({exc}). Re-run ./.venv/Scripts/python.exe train.py.")
        return
    with st.sidebar:
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption(f"**Dataset:** C-MAPSS {meta['subset']}")
        st.caption(f"**RUL model:** {config.REG_NAMES[meta['best_rul_model']]}")
        st.caption(f"**Risk model:** {config.CLF_NAMES[meta['best_clf_model']]}")
        st.caption(f"**Trained:** {meta['trained_at'].replace('T', ' ')}")
        st.caption("Suggestions are model-generated, not engineering diagnoses.")
    try:
        PAGES[page]()
    except Exception as exc:
        st.error(f"This page hit an error: {exc}")
        with st.expander("Details"):
            st.exception(exc)


main()
