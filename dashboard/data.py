"""Cached loaders for trained artefacts, reports and the raw C-MAPSS files."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import load_subset
from src.predict import MaintenancePredictor
from src.utils import load_json

REQUIRED = [config.METADATA_PATH, config.FEATURE_BUILDER_PATH, config.REG_COMPARISON_PATH,
            config.CLF_COMPARISON_PATH, config.FLEET_PATH, config.LAST_CYCLE_PATH]


def ready() -> bool:
    return all(p.exists() for p in REQUIRED)


def version() -> float:
    return config.METADATA_PATH.stat().st_mtime if config.METADATA_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def metadata(_v: float) -> dict:
    return load_json(config.METADATA_PATH)


@st.cache_data(show_spinner=False)
def csv(path: str, _v: float) -> pd.DataFrame | None:
    from pathlib import Path
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def json_report(path: str, _v: float) -> dict:
    from pathlib import Path
    p = Path(path)
    return load_json(p) if p.exists() else {}


@st.cache_data(show_spinner="Loading C-MAPSS files...")
def raw_data(subset: str, _v: float):
    """(train, test) raw sensor data with RUL labels, or (None, None) if files were moved."""
    try:
        return load_subset(subset)
    except FileNotFoundError:
        return None, None


@st.cache_resource(show_spinner="Loading models...")
def predictor(_v: float) -> MaintenancePredictor:
    return MaintenancePredictor()


# convenience wrappers -------------------------------------------------------------------
def fleet(v):
    return csv(str(config.FLEET_PATH), v)


def last_cycle(v):
    return csv(str(config.LAST_CYCLE_PATH), v)


def reg_comparison(v):
    return csv(str(config.REG_COMPARISON_PATH), v)


def clf_comparison(v):
    return csv(str(config.CLF_COMPARISON_PATH), v)


def sensor_importance(v):
    return csv(str(config.SENSOR_IMPORTANCE_PATH), v)
