# Predictive Maintenance AI

An end-to-end machine-learning project for estimating turbofan engine health from
NASA C-MAPSS sensor data. The application predicts remaining useful life (RUL),
estimates the probability of near-term failure, and presents the results in an
interactive Streamlit dashboard.

This is a decision-support demonstration built with simulated data. It is not a
replacement for engineering inspection, maintenance procedures, or safety review.

## What the project does

- Loads and validates the NASA C-MAPSS FD001-FD004 turbofan datasets.
- Builds causal, past-only rolling features from 21 sensor signals.
- Trains RUL regression models: Linear Regression, Random Forest, XGBoost, and
  Support Vector Regression. An LSTM comparison is available as an optional run.
- Trains failure-risk classifiers: Logistic Regression, Random Forest, and XGBoost.
- Uses an engine-level train/validation split and grouped cross-validation to avoid
  leakage between cycles from the same engine.
- Evaluates predictions with regression and classification metrics, including the
  NASA PHM08 score, MAE, RMSE, R-squared, F1, ROC-AUC, and PR-AUC.
- Produces reports, plots, feature importance estimates, and a Streamlit dashboard.

## Project structure

```text
Predictive-Maintenance-AI/
|-- app.py                    # Streamlit entry point
|-- train.py                  # Training and report-generation pipeline
|-- requirements.txt          # Python dependencies
|-- src/                      # Data, features, models, evaluation, and prediction code
|-- dashboard/                # Streamlit pages and presentation components
|-- data/                     # Downloaded C-MAPSS data (not committed)
|-- models/                   # Generated model artifacts (not committed)
|-- reports/                  # Generated metrics, predictions, and figures (not committed)
|-- notebooks/                # Exploratory analysis
|-- tests/                    # Pipeline tests
`-- scripts/                  # Setup and preflight helpers
```

## Requirements

- Python 3.10 to 3.12; Python 3.11 is recommended.
- A CPU-only machine is sufficient.
- The NASA C-MAPSS dataset.

## Setup

Clone the repository and create a virtual environment.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Alternatively, run the setup helper:

```powershell
.\scripts\setup.ps1
```

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS, XGBoost may also require `brew install libomp`.

## Download the data

Download `CMAPSSData.zip` from the
[NASA C-MAPSS dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
Extract the files and copy them into `data/`. For a first run, the following FD001
files are required:

```text
data/
|-- train_FD001.txt
|-- test_FD001.txt
`-- RUL_FD001.txt
```

The complete FD001-FD004 collection can also be placed in this directory. See
[data/README.md](data/README.md) for the file layout and dataset details.

The raw dataset is ignored by Git to keep the repository lightweight and to make
the data source explicit.

## Train the models

Run a quick first pass without hyperparameter tuning:

```powershell
.\.venv\Scripts\python.exe train.py --no-tune
```

Run the default tuned pipeline:

```powershell
.\.venv\Scripts\python.exe train.py
```

Useful options:

```powershell
# Train on a different C-MAPSS subset
.\.venv\Scripts\python.exe train.py --subset FD004

# Use fewer randomized-search iterations
.\.venv\Scripts\python.exe train.py --n-iter 4

# Train the optional LSTM comparison model
.\.venv\Scripts\python.exe -m pip install tensorflow
.\.venv\Scripts\python.exe train.py --lstm
```

Training writes model files to `models/` and metrics, predictions, and figures to
`reports/`. These generated artifacts are ignored by Git and can be recreated.

## Run the dashboard

After training, start Streamlit from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

The dashboard includes:

- Fleet Overview: fleet-wide health and risk summary.
- Engine Inspector: per-engine RUL, risk, sensor trends, and recommendations.
- Sensor Analysis: degradation trends, correlations, and feature importance.
- Manual Prediction: inference for a new reading or uploaded history.
- Model Performance: comparisons, ROC curves, and confusion matrices.
- Data and Methodology: dataset assumptions and modeling notes.

## Validation and tests

Check that the local dependencies and selected dataset are available:

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py --subset FD001
```

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Modeling approach

Each C-MAPSS row represents one engine cycle with three operating settings and 21
sensor readings. The pipeline creates rolling mean, standard deviation, and slope
features using only current and past cycles. It fits preprocessing only on training
engines, uses an engine-wise validation split, and applies `GroupKFold` during
hyperparameter search.

| Assumption | Value |
| --- | --- |
| RUL target | Cycles until end of the training trajectory, capped at 125 |
| Failure-risk label | RUL of 30 cycles or fewer |
| Warning threshold | Predicted RUL of 75 cycles or fewer, or elevated risk |
| Rolling feature window | 10 cycles |
| Reproducibility seed | 42 |

For official test trajectories, evaluation uses the provided RUL value at each
engine's final observed cycle.

## Limitations

- C-MAPSS is a simulated dataset; results do not establish real-world aircraft
  maintenance performance.
- The dashboard's recommendations are model-generated and must be reviewed by
  qualified engineering personnel.
- Performance depends on the selected subset, data quality, and the chosen
  train/validation configuration.

## Further documentation

For the complete setup, troubleshooting, and command sequence, see
[PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Credits

Dataset: A. Saxena, K. Goebel, D. Simon, and N. Eklund, *Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008; NASA
Prognostics Center of Excellence.
