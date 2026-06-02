"""
Train all ML models once. Run from zeroshift root:
  python models/train_models.py
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from utils.data_loader import (
    add_engineered_features,
    lag_features_for_forecast,
    label_failure_in_48h,
    load_raw_dataframe,
    maintenance_feature_matrix,
    train_feature_matrix,
)
from utils.paths import project_root


ROOT = project_root()


def build_autoencoder(timesteps: int, n_features: int) -> keras.Model:
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


def sequences(mat: np.ndarray, timesteps: int) -> np.ndarray:
    xs = []
    for i in range(timesteps - 1, len(mat)):
        xs.append(mat[i - timesteps + 1 : i + 1])
    return np.asarray(xs, dtype=np.float32)


def main() -> None:
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    df0 = load_raw_dataframe()
    df = add_engineered_features(df0)

    # ---------- Model 1: Isolation Forest ----------
    X_if, _ = train_feature_matrix(df)
    y_true = df["anomaly_flag"].values
    contam = float(y_true.sum()) / len(y_true)  # 200 / 8760 — matches injected anomaly rate
    iso = IsolationForest(
        n_estimators=200,
        contamination=contam,
        random_state=42,
    )
    iso.fit(X_if.values)
    pred_if = iso.predict(X_if.values)
    y_pred = (pred_if == -1).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    caught = int(np.sum((y_pred == 1) & (y_true == 1)))
    print("Isolation Forest:")
    print(f"  Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1*100:.1f}%")
    print(f"  Confusion matrix (tn fp; fn tp):\n{cm}")
    print(f"  Anomalies detected correctly: {caught}/200")
    joblib.dump(iso, os.path.join(ROOT, "models", "isolation_forest.pkl"))

    # ---------- Model 2: Gradient Boosting ----------
    dlag = lag_features_for_forecast(df).dropna()
    feature_cols = [
        "lag_1h",
        "lag_2h",
        "lag_3h",
        "lag_6h",
        "lag_24h",
        "rolling_mean_6h",
        "rolling_std_6h",
        "coal_feed_tph",
        "steam_pressure_kgcm2",
        "steam_temp_celsius",
        "bed_temp_celsius",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "shift_encoded",
        "ambient_temp_celsius",
    ]
    # align names with dataframe columns
    for c in feature_cols:
        if c not in dlag.columns:
            raise RuntimeError(f"Missing column {c}")

    X_all = dlag[feature_cols].values
    y_all = dlag["y_next"].values
    split = int(len(dlag) * 0.8)
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    gbr.fit(X_train, y_train)
    pred = gbr.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = mean_absolute_error(y_test, pred)
    acc_proxy = max(0.0, 100.0 * (1.0 - np.mean(np.abs(pred - y_test) / np.maximum(np.abs(y_test), 0.1))))
    print("GB Forecaster:")
    print(f"  R²: {r2:.4f} | RMSE: {rmse:.3f} MW | MAE: {mae:.3f} MW")
    print(f"  Forecast accuracy: {acc_proxy:.1f}%")
    joblib.dump(
        {"model": gbr, "features": feature_cols},
        os.path.join(ROOT, "models", "gb_forecaster.pkl"),
    )

    # ---------- Model 3: Random Forest maintenance ----------
    dm = maintenance_feature_matrix(df)
    feat_rf = [
        "rpm_dev",
        "lube_roll_mean",
        "lube_trend",
        "press_ratio",
        "coal_eff_roll",
        "power_roll_slope",
        "bed_roll_std",
        "coal_efficiency",
        "rolling_mean_power_6h",
        "rolling_std_power_6h",
    ]
    X_rf = dm[feat_rf].fillna(0).values
    y_rf = label_failure_in_48h(df).values
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_rf,
        y_rf,
        test_size=0.2,
        random_state=42,
        stratify=y_rf,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(Xr_train, yr_train)
    proba_raw = rf.predict_proba(Xr_test)
    if proba_raw.shape[1] >= 2:
        proba = proba_raw[:, 1]
    else:
        print(
            "WARNING: RF predict_proba returned a single class column; "
            "using zero failure probability for the test split."
        )
        if len(rf.classes_) == 1 and rf.classes_[0] == 1:
            proba = np.ones(len(Xr_test))
        else:
            proba = np.zeros(len(Xr_test))
    yhat = (proba >= 0.5).astype(int)
    print(
        f"  Train class balance: {int(yr_train.sum())} positive / {len(yr_train)} rows "
        f"({int(yr_train.sum())} pos, {len(yr_train) - int(yr_train.sum())} neg)"
    )
    acc = accuracy_score(yr_test, yhat)
    try:
        aucv = roc_auc_score(yr_test, proba)
    except ValueError:
        aucv = float("nan")
    print("RF Maintenance:")
    print(f"  Accuracy: {acc*100:.1f}% | AUC: {aucv:.3f}")
    print(f"  Failure prediction 48h ahead: {acc*100:.1f}% accuracy")
    joblib.dump({"model": rf, "features": feat_rf}, os.path.join(ROOT, "models", "rf_maintenance.pkl"))

    # ---------- Model 4: LSTM Autoencoder ----------
    feats_ae, _ = train_feature_matrix(df)
    normal_mask = df["anomaly_flag"].values == 0
    scaler = MinMaxScaler()
    scaler.fit(feats_ae.loc[normal_mask].values)
    mat_full = scaler.transform(feats_ae.values)
    timesteps = 24
    X_seq = sequences(mat_full, timesteps)
    # train only on windows ending at normal hour
    end_normal = df["anomaly_flag"].values[timesteps - 1 :] == 0
    Xn = X_seq[end_normal]
    split_ae = int(len(Xn) * 0.9)
    X_tr, X_va = Xn[:split_ae], Xn[split_ae:]
    model = build_autoencoder(timesteps, X_seq.shape[2])
    es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(
        X_tr,
        X_tr,
        validation_data=(X_va, X_va),
        epochs=25,
        batch_size=256,
        verbose=1,
        callbacks=[es],
    )
    pred_tr = model.predict(X_tr, verbose=0)
    mse_normal = np.mean((X_tr - pred_tr) ** 2, axis=(1, 2))
    # anomalies windows
    pred_all = model.predict(X_seq, verbose=0)
    mse_all = np.mean((X_seq - pred_all) ** 2, axis=(1, 2))
    y_end = df["anomaly_flag"].values[timesteps - 1 :]
    mse_anom = mse_all[y_end == 1]
    mse_anom_mean = float(np.nanmean(mse_anom)) if len(mse_anom) else float("nan")
    thr = 0.045
    caught_ae = int(np.sum((mse_all > thr) & (y_end == 1)))
    print("Autoencoder:")
    print(
        f"  Normal MSE: {mse_normal.mean():.3f} (avg) | Anomaly MSE: {mse_anom_mean:.3f} (avg)"
    )
    print(f"  Detection with threshold {thr}: {caught_ae} anomalies caught")
    wpath = os.path.join(ROOT, "models", "autoencoder_weights.weights.h5")
    model.save_weights(wpath)
    joblib.dump(
        {"scaler": scaler, "timesteps": timesteps, "features": list(feats_ae.columns)},
        os.path.join(ROOT, "models", "autoencoder_scaler.pkl"),
    )
    print("Saved models to models/ directory.")


if __name__ == "__main__":
    main()
