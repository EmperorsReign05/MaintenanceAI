"""Check that the local Python environment and selected C-MAPSS dataset are ready."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_PACKAGES = ("numpy", "pandas", "scipy", "sklearn", "xgboost", "streamlit", "joblib")
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local project setup.")
    parser.add_argument("--subset", choices=SUBSETS, default="FD001")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    missing = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if missing:
        print("Missing packages: " + ", ".join(missing))
        print("Fix: python -m pip install -r requirements.txt")
        return 1
    print("Packages: OK")
    from src.data_loader import DatasetNotFoundError, dataset_paths, read_cmapss
    try:
        paths = dataset_paths(args.subset, args.data_dir)
    except DatasetNotFoundError as exc:
        print("Dataset: NOT READY")
        print(exc)
        return 1

    train = read_cmapss(paths["train"])
    test = read_cmapss(paths["test"])
    print(f"Dataset: OK ({args.subset}; train {len(train):,} rows / {train.unit.nunique()} engines; "
          f"test {len(test):,} rows / {test.unit.nunique()} engines)")
    print("Ready to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
