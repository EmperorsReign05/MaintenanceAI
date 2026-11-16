# âš™ï¸ Predictive Maintenance AI
### Intelligent Machine Failure Prediction and Remaining Useful Life Estimation

An end-to-end machine learning system that reads turbofan engine sensor data, estimates **Remaining Useful
Life (RUL)**, predicts the **probability of failure within the next 30 cycles**, assigns a **health status**, and
explains which sensors drive degradation, all inside an industrial monitoring dashboard. Runs on a CPU-only laptop.

> Maintenance suggestions in this app are **model-generated decision support**, not engineering diagnoses.
> Real maintenance decisions require qualified engineers and physical inspection.

---

## Highlights

- **Authentic data**: NASA C-MAPSS turbofan degradation simulation (FD001-FD004). No fabricated data or scores.
- **No temporal leakage**: engine-wise splits, past-only rolling features (verified by a unit test), scaler fitted on
  training engines only, engine-grouped cross-validation.
- **Two prediction tasks**: RUL regression (Linear Regression, Random Forest, XGBoost, SVR, optional LSTM) and
  failure-risk classification (Logistic Regression, Random Forest, XGBoost).
- **Proper evaluation**: MAE, RMSE, RÂ², NASA PHM08 score; Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix.
- **Explainability**: grouped permutation importance per sensor, degradation patterns, healthy vs degrading comparison.
- **Dashboard**: fleet KPIs, engine inspector with sensor cards and trends, manual prediction through the same saved
  pipeline, model comparison, and maintenance suggestions.
- **Reproducible**: fixed seeds (42) for NumPy, Python, scikit-learn, XGBoost and TensorFlow.

---

## Architecture

```
NASA C-MAPSS text files (train / test / RUL)
        â”‚
        â–¼
data_loader â”€ parse 26 columns â”€ validate â”€ missing-value report â”€ RUL labels (capped at 125)
        â”‚
        â–¼
split by ENGINE (80% train / 20% validation)          official test engines kept separate
        â”‚
        â–¼
FeatureBuilder (fit on training engines only, saved to models/feature_builder.joblib)
  drop constant sensors â†’ [FD002/FD004: per-operating-condition normalisation]
  â†’ rolling mean / std / slope over the last 10 cycles (past only) â†’ StandardScaler
        â”‚
        â”œâ”€â”€â–º RUL regressors   (GroupKFold tuning, select by validation RMSE)
        â””â”€â”€â–º failure classifiers (GroupKFold tuning, threshold = max F1 on validation, select by PR-AUC)
        â”‚
        â–¼
evaluation on test engines â”€â–º reports/ (CSV, JSON, figures)   models/ (joblib + metadata)
        â”‚
        â–¼
Streamlit dashboard (app.py) â”€â–º MaintenancePredictor (same FeatureBuilder) â”€â–º status + suggestions
```

## Project structure

```
Predictive-Maintenance-AI/
â”œâ”€â”€ app.py                        # .venv/Scripts/python.exe -m streamlit run app.py
â”œâ”€â”€ train.py                      # .venv/Scripts/python.exe train.py
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â”œâ”€â”€ .streamlit/config.toml        # charcoal/blue theme
â”œâ”€â”€ data/                         # C-MAPSS .txt files go here (see data/README.md)
â”œâ”€â”€ notebooks/01_eda.ipynb        # exploratory analysis
â”œâ”€â”€ models/                       # generated: models, feature builder, metadata
â”œâ”€â”€ reports/                      # generated: metrics, comparisons, predictions, figures
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ config.py                 # paths, assumptions (RUL cap, failure window), seeds
â”‚   â”œâ”€â”€ data_loader.py            # parse, validate, label RUL
â”‚   â”œâ”€â”€ features.py               # FeatureBuilder + engine-wise split
â”‚   â”œâ”€â”€ eda.py                    # EDA figures and summary
â”‚   â”œâ”€â”€ models.py                 # model builders + search spaces
â”‚   â”œâ”€â”€ lstm.py                   # optional LSTM (TensorFlow, CPU)
â”‚   â”œâ”€â”€ evaluate.py               # metrics and comparison plots
â”‚   â”œâ”€â”€ insights.py               # health status, sensor importance, recommendations
â”‚   â”œâ”€â”€ predict.py                # MaintenancePredictor for new data
â”‚   â””â”€â”€ utils.py
â”œâ”€â”€ dashboard/
â”‚   â”œâ”€â”€ theme.py  data.py  components.py
â”‚   â””â”€â”€ views/  fleet.py  engine.py  sensors.py  manual.py  performance.py  about.py
â””â”€â”€ tests/test_pipeline.py
```

---

## Setup and run

Python **3.10-3.12** (3.11 recommended). No GPU needed.

The maintained setup, data, run, and troubleshooting instructions are in
[`PROJECT_GUIDE.md`](PROJECT_GUIDE.md). The commands below use the project's virtual-environment
interpreter so the correct dependencies are always selected.

**Windows (PowerShell)**
Use the single ordered command block in [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md#run-this-project-exact-ordered-commands).

**macOS / Linux**
```bash
cd Predictive-Maintenance-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(macOS: `brew install libomp` for XGBoost.)

**Get the data** - download `CMAPSSData.zip` from
https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data and copy `train_FD001.txt`, `test_FD001.txt`,
`RUL_FD001.txt` into `data/`. Details and file format: [`data/README.md`](data/README.md).

**Train**
```bash
.venv/Scripts/python.exe train.py --no-tune        # Windows quick first run (~1-3 min)
.venv/Scripts/python.exe train.py                  # Windows tuned run (~10-25 min)
.venv/Scripts/python.exe train.py --subset FD004   # harder subset: 6 operating conditions, 2 fault modes
.venv/Scripts/python.exe -m pip install tensorflow
.venv/Scripts/python.exe train.py --lstm           # also train the optional LSTM (CPU, ~5-15 min)
```

**Dashboard**
```bash
.venv/Scripts/python.exe -m streamlit run app.py
```
â†’ http://localhost:8501

**Tests**
```bash
.venv/Scripts/python.exe -m pytest -q
```

---

## Dataset and assumptions

C-MAPSS simulates turbofan engines that start with unknown manufacturing variation and wear, then develop a fault
that grows until a failure threshold is reached. Each row is one cycle: engine id, cycle, 3 operating settings and
21 sensors. Training trajectories run to failure; test trajectories stop earlier, and `RUL_FDxxx.txt` gives the true
remaining cycles at their last observed cycle.

| Assumption | Value | Why |
|---|---|---|
| End of life | last cycle of a training trajectory | the simulation's failure threshold |
| RUL label | cycles to end of life, **capped at 125** | early life shows no measurable degradation; the piecewise-linear target is standard for C-MAPSS |
| Failure-risk class | **RUL â‰¤ 30 cycles** | a maintenance-planning horizon |
| Warning status | predicted RUL â‰¤ 75 or risk score â‰¥ half the threshold | early heads-up band |
| High-risk status | risk score â‰¥ threshold or predicted RUL â‰¤ 30 | needs attention |

Test RUL metrics use NASA's ground-truth values at each test engine's **last cycle** (the standard protocol).

## Algorithms

| Model | Purpose | How it works |
|---|---|---|
| Linear Regression | baseline | weighted sum of features; shows how much non-linearity matters |
| Random Forest | RUL + risk | many decision trees on bootstrap samples, averaged; robust to noisy sensors |
| XGBoost | RUL + risk | gradient-boosted trees added sequentially, each correcting remaining error |
| SVR (RBF) | RUL | smooth kernel function with an error-tolerant epsilon tube; trained on a 6,000-row subsample (O(nÂ²)) |
| LSTM (optional) | RUL | recurrent network reading the last 30 cycles in order; compared in reports, not used for live predictions |
| Logistic Regression | risk | class-weighted linear classifier for the minority "at risk" class |
| Permutation importance | explanation | shuffles all features of one sensor and measures how much RUL error grows |

## Dashboard pages

| Page | Contents |
|---|---|
| Fleet Overview | engines monitored, healthy / warning / high-risk counts, RUL per engine, status donut, predicted vs actual RUL, risk distribution, fleet register |
| Engine Inspector | choose an engine: RUL and risk gauges, health badge, sensor cards (drift vs early life), RUL and risk trends, sensor history charts, maintenance suggestion |
| Sensor Analysis | degradation pattern of any sensor across the fleet, healthy vs degrading distributions, correlation with RUL, feature importance |
| Manual Prediction | enter a reading (appended to a real engine's history or alone), range checks, prediction via the saved pipeline; or upload a history CSV/TXT |
| Model Performance | regression and classification tables, bar charts, predicted-vs-actual scatter, ROC curves, confusion matrices |
| Data & Methodology | dataset, assumptions, pipeline, algorithm notes, EDA figures, limitations |

## Results

Copy your numbers from `reports/regression_comparison.csv` and `reports/classification_comparison.csv` after training.
Nothing here is pre-filled; every metric is produced by your own run.

| RUL model | RMSE | MAE | RÂ² |
|---|---|---|---|
| *from reports/regression_comparison.csv* | | | |

| Failure classifier | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| *from reports/classification_comparison.csv* | | | | |

## Troubleshooting

| Problem | Fix |
|---|---|
| `C-MAPSS files for FD001 were not found` | the three .txt files must be inside `data/` (any sub-folder is fine) |
| `ModuleNotFoundError: src` | run commands from the project root (where `app.py` is) |
| Dashboard shows the setup screen | run `.venv/Scripts/python.exe train.py` first, then refresh |
| `expected 26 columns` | use the original NASA .txt files, not an edited CSV |
| XGBoost `libomp` error (macOS) | `brew install libomp` |
| TensorFlow will not install | it is optional; skip `--lstm` or use Python 3.11 |
| Training too slow | `--no-tune` or `--n-iter 4` |
| Streamlit `width` argument error | `pip install -U "streamlit>=1.50"` |

## Limitations

C-MAPSS is simulated, so models learn simulated degradation physics. Test metrics on FD001 rest on 100 engines,
so last-cycle classification metrics move in large steps. A real deployment would need calibration on fleet data,
drift monitoring and engineering validation.

## Credits

Dataset: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", PHM 2008; NASA Prognostics Center of Excellence.

