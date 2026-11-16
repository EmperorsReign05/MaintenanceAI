# Predictive Maintenance AI: Project Guide

## Purpose

This project predicts turbofan-engine Remaining Useful Life (RUL) and failure risk within the next 30 cycles from NASA C-MAPSS simulated run-to-failure data. Results are presented in a Streamlit dashboard. Model recommendations are decision support, not engineering diagnoses.

Live deployment: [predictive-maintenance-ai-proj.streamlit.app](https://predictive-maintenance-ai-proj.streamlit.app/)

## Included capabilities

- Validated C-MAPSS loading for FD001-FD004.
- Engine-level splits and causal rolling sensor features that prevent leakage.
- RUL regression (linear regression, random forest, XGBoost, and SVR).
- Failure-risk classification (logistic regression, random forest, and XGBoost).
- Evaluation reports, figures, feature importance, persisted artifacts, dashboard, and tests.

## Requirements

- CPython 3.10-3.12; 3.11 is recommended.
- Internet access to install packages and download the NASA source data.
- No GPU is needed. TensorFlow is optional for the LSTM experiment.
Run these in order:

# 1. Install project dependencies
.\scripts\setup.ps1

# 2. Download NASA CMAPSSData.zip, extract it, then place these in data\:
# train_FD001.txt
# test_FD001.txt
# RUL_FD001.txt

# 3. Verify packages and data
.\.venv\Scripts\python.exe scripts\preflight.py

# 4. Train a quick first model
.\.venv\Scripts\python.exe train.py --no-tune

# 5. Start dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py

For full tuned training, use this instead of step 4:

.\.venv\Scripts\python.exe train.py
## Setup on Windows PowerShell

Run from the project root. This creates `.venv` and installs dependencies:

```powershell
.\scripts\setup.ps1
```

If local scripts are blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1
```

The script uses `py -3.11` by default. Select another installed interpreter when needed:

```powershell
.\scripts\setup.ps1 -Python "py -3.12"
```

Install the optional LSTM dependency during setup:

```powershell
.\scripts\setup.ps1 -IncludeLstm
```

Manual equivalent:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dataset setup

Download `CMAPSSData.zip` from [NASA's C-MAPSS dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data), extract it, and put these original files anywhere below `data/`:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

Run the preflight check before training:

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py
```

For another subset, for example FD004:

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py --subset FD004
```

## Current run commands

| Task | Command |
| --- | --- |
| Check environment and data | `.\.venv\Scripts\python.exe scripts\preflight.py` |
| Fast first FD001 run | `.\.venv\Scripts\python.exe train.py --no-tune` |
| Full tuned FD001 run | `.\.venv\Scripts\python.exe train.py` |
| Fast FD004 run | `.\.venv\Scripts\python.exe train.py --subset FD004 --no-tune` |
| Reduced tuning | `.\.venv\Scripts\python.exe train.py --n-iter 4` |
| Fast run without EDA figures | `.\.venv\Scripts\python.exe train.py --no-tune --skip-eda` |
| Optional LSTM | `.\.venv\Scripts\python.exe -m pip install tensorflow` then `.\.venv\Scripts\python.exe train.py --lstm --no-tune` |
| Dashboard | `.\.venv\Scripts\python.exe -m streamlit run app.py` |
| Tests | `.\.venv\Scripts\python.exe -m pytest -q` |

The local dashboard normally opens at `http://localhost:8501`. The deployed dashboard is available at [predictive-maintenance-ai-proj.streamlit.app](https://predictive-maintenance-ai-proj.streamlit.app/).

## Workflow

```text
NASA C-MAPSS files
  -> validation and RUL labels
  -> engine-wise train/validation split
  -> FeatureBuilder fitted only on training engines
  -> RUL and failure-risk models
  -> reports, artifacts, and dashboard
```

Rolling features use only current and prior cycles. Scaling and operating-condition normalization are fitted only on training engines, so validation and test data do not leak into preprocessing.

## Outputs after training

| Location | Contents |
| --- | --- |
| `models/feature_builder.joblib` | fitted preprocessing pipeline |
| `models/rul_*.joblib` | RUL regression models |
| `models/clf_*.joblib` | failure-risk classifiers |
| `models/metadata.json` | selected models, assumptions, and run metadata |
| `reports/regression_comparison.csv` | RUL comparison |
| `reports/classification_comparison.csv` | classifier comparison and thresholds |
| `reports/fleet_predictions.csv` | every test-engine cycle prediction |
| `reports/test_last_cycle.csv` | official last-cycle test results |
| `reports/figures/` | EDA and evaluation visualizations |

## Project layout

```text
app.py                 Streamlit entry point
train.py               training and report-generation pipeline
scripts/setup.ps1      environment setup
scripts/preflight.py   dependency and data validation
src/                   loading, features, models, metrics, inference
dashboard/             Streamlit pages and components
tests/                 unit tests
data/                  NASA source files (not committed)
models/                trained artifacts (not committed)
reports/               generated reports and figures (not committed)
```

## Troubleshooting and limitations

- If `py` is not recognized, install CPython from python.org and select **Add Python to PATH**, or pass the interpreter with `-Python`.
- If Python lacks `pip`, use supported CPython rather than a minimal/MSYS Python build.
- If preflight cannot find data, ensure all three source files for the selected subset are below `data/` and retain their names.
- If the dashboard shows setup status, finish a successful training run first.
- Start with `--no-tune` or `--n-iter 4` if training is slow.

RUL reports MAE, RMSE, R-squared, and NASA's PHM08 score. Classification reports precision, recall, F1, ROC-AUC, PR-AUC, and a validation-selected F1 threshold. C-MAPSS is simulated; deployment needs real-fleet calibration, drift monitoring, safeguards, and engineering validation.
