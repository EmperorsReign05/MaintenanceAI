"""Dataset, assumptions, algorithms and EDA figures."""
from __future__ import annotations

import streamlit as st

from src import config

from .. import data
from ..theme import header, note


def render() -> None:
    v = data.version()
    meta = data.metadata(v)
    a, e = meta["assumptions"], meta["engines"]
    header("Data & Methodology", "Dataset, labelling assumptions and how each model works")

    c = st.columns(4)
    c[0].metric("Subset", meta["subset"])
    c[1].metric("Engines (train / val / test)", f"{e['train']} / {e['validation']} / {e['test']}")
    c[2].metric("Sensors used", f"{len(meta['sensors_used'])} of 21")
    c[3].metric("Engineered features", meta["n_features"])

    st.markdown(f"""
### Dataset
[NASA C-MAPSS Turbofan Engine Degradation Simulation]({config.DATASET_URL}) - simulated run-to-failure data
for a fleet of turbofan engines. Each row is one operating cycle: engine id, cycle number, 3 operating settings and
21 sensor measurements. Training engines run until failure; test engines stop earlier, and NASA supplies their
true RUL at the last recorded cycle. Missing values found: train {meta['missing_values']['train']['total_missing']},
test {meta['missing_values']['test']['total_missing']}.

### Assumptions
- **End of life** = the last cycle of a training trajectory (the simulation's failure threshold).
- **RUL label** = cycles until that point, **capped at {a['rul_cap']}** (piecewise-linear): early in life an engine
  shows no measurable degradation, so a constant 'healthy' target is more learnable than a long linear ramp.
- **Failure-risk class** = RUL &le; **{a['failure_window']}** cycles.
- **Health status**: High risk = classifier score &ge; threshold or predicted RUL &le; {a['failure_window']};
  Warning = predicted RUL &le; {a['warning_rul']} or score &ge; half the threshold; otherwise Healthy.

### Pipeline (no temporal leakage)
1. Split **by engine** (80/20) so no engine contributes cycles to both train and validation; the official test set is separate.
2. Drop sensors that are constant in training ({', '.join(meta['sensors_dropped']) or 'none'}).
3. Features use only current and **past** cycles: raw value, rolling mean, rolling std and rolling slope
   over {a['rolling_window']} cycles, plus cycle number. Multi-condition subsets are first normalised per operating condition.
4. StandardScaler fitted on training engines only; the identical saved object scores dashboard inputs.
5. Hyper-parameters tuned with engine-grouped cross-validation ({meta['tuning']['cv']}); seed {meta['seed']}.

### Algorithms
- **Linear Regression** - straight-line baseline; shows how much non-linearity the others add.
- **Random Forest** - averages many decision trees trained on bootstrap samples; captures non-linear wear and is robust to sensor noise.
- **XGBoost** - gradient boosting: trees are added one at a time, each fixing the remaining errors; usually the strongest tabular model.
- **Support Vector Regression** - fits a smooth RBF-kernel function that ignores errors smaller than epsilon; trained on a
  {config.SVR_MAX_ROWS:,}-row subsample because its cost grows quadratically with rows.
- **LSTM** (optional) - a recurrent neural network that reads the last {config.LSTM_SEQUENCE} cycles in order. Compared in the
  performance table; the app uses the selected tabular model for live predictions.
- **Failure classifiers** - Logistic Regression, Random Forest and XGBoost with class weighting for the minority at-risk class.

Trained {meta['trained_at']}.
""")
    figs = sorted(config.FIGURES_DIR.glob("0[1-6]_*.png"))
    if figs:
        st.subheader("Exploratory data analysis")
        cols = st.columns(2)
        for i, p in enumerate(figs):
            cols[i % 2].image(str(p), width="stretch")
    note("<b>Limitations.</b> C-MAPSS is simulated data. Predictions and suggestions from this system are decision "
         "support for planning and are not engineering diagnoses; real maintenance decisions require qualified "
         "engineers, inspections and the manufacturer's procedures.", warn=True)
