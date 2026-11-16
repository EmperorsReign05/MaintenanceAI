"""Optional LSTM RUL model (TensorFlow/Keras, CPU-friendly). Enabled with: ./.venv/Scripts/python.exe train.py --lstm

An LSTM (Long Short-Term Memory network) reads the last LSTM_SEQUENCE cycles of sensor readings in order
and learns how their temporal pattern maps to remaining life. Engines with fewer cycles than the window are
left-padded by repeating their first reading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger, is_available

log = get_logger()
HAS_TF = is_available("tensorflow")


def sequence_columns(feature_names: list[str]) -> list[str]:
    return [c for c in feature_names if c.endswith("_raw")] + ["cycle"]


def make_sequences(X: pd.DataFrame, units: np.ndarray, seq_len: int = config.LSTM_SEQUENCE) -> np.ndarray:
    """One sequence per row, ending at that row (past cycles only). X and units must be sorted by unit, cycle."""
    from numpy.lib.stride_tricks import sliding_window_view

    arr = X.to_numpy(dtype=np.float32)
    out = np.empty((len(arr), seq_len, arr.shape[1]), dtype=np.float32)
    start = 0
    for u in pd.unique(units):
        n = int((units == u).sum())
        block = arr[start:start + n]
        padded = np.concatenate([np.repeat(block[:1], seq_len - 1, axis=0), block])
        out[start:start + n] = sliding_window_view(padded, (seq_len, arr.shape[1]))[:, 0]
        start += n
    return out


def train_lstm(X_tr: np.ndarray, y_tr, X_va: np.ndarray, y_va, epochs: int = 40):
    if not HAS_TF:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
    import tensorflow as tf

    tf.random.set_seed(config.RANDOM_STATE)
    tf.config.experimental.enable_op_determinism()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X_tr.shape[1:]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=6, restore_best_weights=True)
    model.fit(X_tr, np.asarray(y_tr, np.float32), validation_data=(X_va, np.asarray(y_va, np.float32)),
              epochs=epochs, batch_size=256, callbacks=[stop], verbose=2)
    model.save(config.LSTM_PATH)
    return model


def predict_lstm(model, X_seq: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(X_seq, batch_size=1024, verbose=0).ravel(), 0, config.RUL_CAP)
