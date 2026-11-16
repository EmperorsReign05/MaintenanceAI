"""Central configuration: paths, dataset layout, modelling assumptions and seeds."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_URL = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
DATASET_ZIP_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
DEFAULT_SUBSET = "FD001"
MULTI_CONDITION_SUBSETS = {"FD002", "FD004"}   # six operating conditions -> condition-wise normalisation

# C-MAPSS text files: 26 space-separated numeric columns, no header
OP_SETTINGS = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OP_SETTINGS, *SENSORS]

SENSOR_DESCRIPTIONS = {
    "s1": "Fan inlet temperature (T2)", "s2": "LPC outlet temperature (T24)",
    "s3": "HPC outlet temperature (T30)", "s4": "LPT outlet temperature (T50)",
    "s5": "Fan inlet pressure (P2)", "s6": "Bypass-duct pressure (P15)",
    "s7": "HPC outlet pressure (P30)", "s8": "Physical fan speed (Nf)",
    "s9": "Physical core speed (Nc)", "s10": "Engine pressure ratio (epr)",
    "s11": "HPC outlet static pressure (Ps30)", "s12": "Fuel flow / Ps30 (phi)",
    "s13": "Corrected fan speed (NRf)", "s14": "Corrected core speed (NRc)",
    "s15": "Bypass ratio (BPR)", "s16": "Burner fuel-air ratio (farB)",
    "s17": "Bleed enthalpy (htBleed)", "s18": "Demanded fan speed (Nf_dmd)",
    "s19": "Demanded corrected fan speed (PCNfR_dmd)", "s20": "HPT coolant bleed (W31)",
    "s21": "LPT coolant bleed (W32)",
}

# ---- Modelling assumptions (documented in README) ---------------------------------------
RUL_CAP = 125            # piecewise-linear RUL: early life treated as 'healthy plateau'
FAILURE_WINDOW = 30      # positive class: engine will reach end-of-life within 30 cycles
WARNING_RUL = 75         # RUL at or below this (and above FAILURE_WINDOW) -> Warning
ROLLING_WINDOW = 10      # cycles of history used for rolling features (past-only, causal)
LSTM_SEQUENCE = 30       # cycles per LSTM input sequence
SVR_MAX_ROWS = 6000      # SVR is O(n^2): train on a fixed random subsample
VAL_ENGINE_FRACTION = 0.2
RANDOM_STATE = 42

# ---- Artefacts -----------------------------------------------------------------------------
FEATURE_BUILDER_PATH = MODELS_DIR / "feature_builder.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
LSTM_PATH = MODELS_DIR / "lstm_rul.keras"
METRICS_PATH = REPORTS_DIR / "metrics.json"
REG_COMPARISON_PATH = REPORTS_DIR / "regression_comparison.csv"
CLF_COMPARISON_PATH = REPORTS_DIR / "classification_comparison.csv"
ROC_PATH = REPORTS_DIR / "roc_curves.json"
FLEET_PATH = REPORTS_DIR / "fleet_predictions.csv"          # every cycle of every test engine
LAST_CYCLE_PATH = REPORTS_DIR / "test_last_cycle.csv"        # one row per test engine, all models
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
SENSOR_IMPORTANCE_PATH = REPORTS_DIR / "sensor_importance.csv"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"

REG_NAMES = {"linear_regression": "Linear Regression", "random_forest": "Random Forest",
             "xgboost": "XGBoost", "svr": "Support Vector Regression", "lstm": "LSTM"}
CLF_NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
             "xgboost": "XGBoost"}


def reg_path(key: str) -> Path:
    return MODELS_DIR / f"rul_{key}.joblib"


def clf_path(key: str) -> Path:
    return MODELS_DIR / f"clf_{key}.joblib"


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
