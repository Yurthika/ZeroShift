"""Two-layer anomaly detection: Isolation Forest + optional LSTM autoencoder."""
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from utils.data_loader import train_feature_matrix
from utils.paths import project_root

try:
    from tensorflow import keras
    from tensorflow.keras import layers

    KERAS_AVAILABLE = True
except Exception:
    keras = None
    layers = None
    KERAS_AVAILABLE = False


def build_autoencoder(timesteps: int, n_features: int):
    if not KERAS_AVAILABLE:
        raise RuntimeError("Keras is not available")
    inp = keras.Input(shape=(timesteps, n_features))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.RepeatVector(timesteps)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n_features))(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


@st.cache_resource(show_spinner=False)
def load_isolation_forest():
    path = os.path.join(project_root(), "models", "isolation_forest.pkl")
    if os.path.isfile(path):
        return joblib.load(path)
    from utils.data_loader import load_processed_data

    df = load_processed_data()
    X, _ = train_feature_matrix(df)
    y = df["anomaly_flag"].values
    model = IsolationForest(
        n_estimators=200,
        contamination=float(max(int(y.sum()), 1)) / len(y),
        random_state=42,
    )
    model.fit(X.values)
    return model


@st.cache_resource(show_spinner=False)
def load_autoencoder_bundle():
    if not KERAS_AVAILABLE:
        raise RuntimeError("Keras is not available")
    root = project_root()
    wpath = os.path.join(root, "models", "autoencoder_weights.weights.h5")
    if not os.path.isfile(wpath):
        wpath = os.path.join(root, "models", "autoencoder_weights.h5")
    spath = os.path.join(root, "models", "autoencoder_scaler.pkl")
    if not os.path.isfile(spath) or not os.path.isfile(wpath):
        raise FileNotFoundError("Autoencoder weights or scaler not found")
    meta = joblib.load(spath)
    scaler: MinMaxScaler = meta["scaler"]
    timesteps = int(meta["timesteps"])
    features: List[str] = meta["features"]
    model = build_autoencoder(timesteps, len(features))
    model.load_weights(wpath)
    return model, scaler, timesteps, features


def _sequences(mat: np.ndarray, timesteps: int) -> np.ndarray:
    xs = []
    for i in range(timesteps - 1, len(mat)):
        xs.append(mat[i - timesteps + 1 : i + 1])
    return np.asarray(xs, dtype=np.float32)


def _layer2_sklearn_scores(df: pd.DataFrame) -> pd.Series:
    """Fallback Layer-2 when Keras/autoencoder is unavailable."""
    X, _ = train_feature_matrix(df)
    roll = X.rolling(24, min_periods=1).mean().bfill().ffill().values
    flags = df["anomaly_flag"].values
    contam = float(max(int(flags.sum()), 1)) / len(flags)
    model = IsolationForest(n_estimators=200, contamination=contam, random_state=7)
    model.fit(roll)
    return pd.Series(-model.score_samples(roll), index=df.index, name="layer2_score")


def compute_reconstruction_errors(df: pd.DataFrame) -> pd.Series:
    if KERAS_AVAILABLE:
        try:
            model, scaler, timesteps, feats = load_autoencoder_bundle()
            sub = df[feats].copy().ffill().bfill()
            mat = scaler.transform(sub.values)
            seq = _sequences(mat, timesteps)
            if len(seq) == 0:
                return _layer2_sklearn_scores(df)
            pred = model.predict(seq, verbose=0)
            mse = np.mean((seq - pred) ** 2, axis=(1, 2))
            idx = df.index[timesteps - 1 :]
            out = pd.Series(mse, index=idx)
            return out.reindex(df.index).bfill().fillna(0.0)
        except Exception:
            pass
    return _layer2_sklearn_scores(df)


def run_detection(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    clf = load_isolation_forest()
    X, _ = train_feature_matrix(out)
    pred1 = clf.predict(X.values)
    scores1 = -clf.score_samples(X.values)
    out["iforest_score"] = scores1
    out["layer1_anomaly"] = (pred1 == -1).astype(int)

    err = compute_reconstruction_errors(out)
    out["ae_mse"] = err
    if KERAS_AVAILABLE:
        threshold = 0.045
    else:
        threshold = float(np.quantile(err.values, 0.977))
    out["layer2_anomaly"] = (out["ae_mse"] > threshold).astype(int)
    out["combined_anomaly"] = ((out["layer1_anomaly"] + out["layer2_anomaly"]) >= 1).astype(int)
    return out


def layer_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["anomaly_flag"].values
    y1 = df["layer1_anomaly"].values
    tp = int(np.sum((y1 == 1) & (y_true == 1)))
    fp = int(np.sum((y1 == 1) & (y_true == 0)))
    fn = int(np.sum((y1 == 0) & (y_true == 1)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp}
