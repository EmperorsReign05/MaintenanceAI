"""Loading, validating and labelling the NASA C-MAPSS turbofan dataset.

Each subset FDxxx has three files:
  train_FDxxx.txt  full run-to-failure trajectories (the last cycle of each unit is its failure point)
  test_FDxxx.txt   trajectories cut off at an unknown point before failure
  RUL_FDxxx.txt    true remaining cycles at the last test cycle of each unit (one line per unit)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

log = get_logger()


class DatasetNotFoundError(FileNotFoundError):
    pass


def download_help(subset: str) -> str:
    return f"""
C-MAPSS files for {subset} were not found under: {config.DATA_DIR}

This project trains only on the authentic NASA dataset. To get it:
  1. Open {config.DATASET_URL}
     (direct link: {config.DATASET_ZIP_URL})
  2. Download CMAPSSData.zip and extract it.
  3. Copy train_{subset}.txt, test_{subset}.txt and RUL_{subset}.txt into the  data/  folder
     (copying the whole extracted folder into data/ also works - files are found recursively).

Then run:  ./.venv/Scripts/python.exe train.py --subset {subset}
"""


def _find(name: str, data_dir: Path) -> Path | None:
    direct = data_dir / name
    if direct.exists():
        return direct
    hits = sorted(data_dir.rglob(name)) if data_dir.exists() else []
    return hits[0] if hits else None


def dataset_paths(subset: str = config.DEFAULT_SUBSET, data_dir: Path = config.DATA_DIR) -> dict[str, Path]:
    if subset not in config.SUBSETS:
        raise ValueError(f"subset must be one of {config.SUBSETS}")
    paths = {kind: _find(f"{kind}_{subset}.txt", Path(data_dir)) for kind in ("train", "test", "RUL")}
    if any(p is None for p in paths.values()):
        raise DatasetNotFoundError(download_help(subset))
    return paths


def read_cmapss(path: Path) -> pd.DataFrame:
    """Read one whitespace-separated C-MAPSS trajectory file."""
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(config.COLUMNS):
        raise ValueError(f"{path.name}: expected {len(config.COLUMNS)} columns, found {df.shape[1]}. "
                         "Is this an original C-MAPSS file?")
    df.columns = config.COLUMNS
    validate(df, path.name)
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def validate(df: pd.DataFrame, name: str = "data") -> None:
    if df.isna().any().any():
        raise ValueError(f"{name}: contains missing values in {df.columns[df.isna().any()].tolist()}")
    if not np.all(np.isfinite(df.to_numpy(dtype=float))):
        raise ValueError(f"{name}: contains non-finite values")
    gaps = df.groupby("unit")["cycle"].apply(lambda c: (np.diff(np.sort(c.to_numpy())) != 1).any())
    if gaps.any():
        log.warning("%s: %d units have non-consecutive cycles", name, int(gaps.sum()))


def missing_report(df: pd.DataFrame) -> dict:
    return {"total_missing": int(df.isna().sum().sum()),
            "by_column": {c: int(v) for c, v in df.isna().sum().items() if v}}


def add_train_rul(df: pd.DataFrame, cap: int = config.RUL_CAP) -> pd.DataFrame:
    """Training engines run to failure: RUL = last cycle - current cycle, capped (piecewise linear)."""
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul_true"] = last - df["cycle"]
    df["rul"] = df["rul_true"].clip(upper=cap)
    return df


def add_test_rul(test: pd.DataFrame, rul_end: pd.Series, cap: int = config.RUL_CAP) -> pd.DataFrame:
    """Test engines stop early: RUL at a cycle = RUL at last cycle + (last cycle - cycle)."""
    test = test.copy()
    units = np.sort(test["unit"].unique())
    if len(units) != len(rul_end):
        raise ValueError(f"RUL file has {len(rul_end)} rows but test set has {len(units)} engines")
    end_map = dict(zip(units, rul_end.to_numpy()))
    last = test.groupby("unit")["cycle"].transform("max")
    test["rul_true"] = test["unit"].map(end_map) + (last - test["cycle"])
    test["rul"] = test["rul_true"].clip(upper=cap)
    return test


def load_subset(subset: str = config.DEFAULT_SUBSET, data_dir: Path = config.DATA_DIR):
    """Return (train_df, test_df) with RUL labels attached."""
    paths = dataset_paths(subset, data_dir)
    train = read_cmapss(paths["train"])
    test = read_cmapss(paths["test"])
    rul_end = pd.read_csv(paths["RUL"], sep=r"\s+", header=None, engine="python").iloc[:, 0]
    train, test = add_train_rul(train), add_test_rul(test, rul_end)
    log.info("%s: train %s rows / %d engines | test %s rows / %d engines", subset,
             f"{len(train):,}", train.unit.nunique(), f"{len(test):,}", test.unit.nunique())
    return train, test


def label_failure(rul: pd.Series | np.ndarray, window: int = config.FAILURE_WINDOW) -> np.ndarray:
    return (np.asarray(rul) <= window).astype(int)
