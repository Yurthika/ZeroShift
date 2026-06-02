"""Gradient Boosting forecasting utilities."""
import os
from collections import deque

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import add_engineered_features, lag_features_for_forecast
from utils.paths import project_root


@st.cache_resource(show_spinner=False)
def load_gb_forecaster():
    path = os.path.join(project_root(), "models", "gb_forecaster.pkl")
    return joblib.load(path)


def prepare_forecast_base(df: pd.DataFrame) -> pd.DataFrame:
    d0 = add_engineered_features(df)
    return lag_features_for_forecast(d0)


def forecast_next_hours(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    bundle = load_gb_forecaster()
    model = bundle["model"]
    feature_cols = bundle["features"]

    base = prepare_forecast_base(df).dropna()
    last_ts = base.index[-1]
    last_row = base.iloc[-1].copy()

    net_hist = deque(base["net_power_mw"].tail(48).tolist(), maxlen=96)

    preds = []
    cur_ts = last_ts
    for h in range(1, hours + 1):
        cur_ts = cur_ts + pd.Timedelta(hours=1)
        lag_1h = net_hist[-1]
        lag_2h = net_hist[-2] if len(net_hist) >= 2 else lag_1h
        lag_3h = net_hist[-3] if len(net_hist) >= 3 else lag_1h
        lag_6h = net_hist[-6] if len(net_hist) >= 6 else lag_1h
        lag_24h = net_hist[-24] if len(net_hist) >= 24 else lag_1h
        tail = list(net_hist)[-6:]
        rolling_mean_6h = float(np.mean(tail))
        rolling_std_6h = float(np.std(tail)) if len(tail) > 1 else 0.0

        row = last_row.copy()
        row["lag_1h"] = lag_1h
        row["lag_2h"] = lag_2h
        row["lag_3h"] = lag_3h
        row["lag_6h"] = lag_6h
        row["lag_24h"] = lag_24h
        row["rolling_mean_6h"] = rolling_mean_6h
        row["rolling_std_6h"] = rolling_std_6h
        row["hour_of_day"] = cur_ts.hour
        row["day_of_week"] = cur_ts.dayofweek
        row["is_weekend"] = cur_ts.dayofweek >= 5
        # shift label from hour
        if 6 <= cur_ts.hour < 14:
            sh = "Morning"
            enc = 0
        elif 14 <= cur_ts.hour < 22:
            sh = "Afternoon"
            enc = 1
        else:
            sh = "Night"
            enc = 2
        row["shift"] = sh
        row["shift_encoded"] = enc

        x = np.asarray(row[feature_cols].values, dtype=float).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append({"timestamp": cur_ts, "predicted_net_mw": yhat, "shift": sh})
        net_hist.append(yhat)

    return pd.DataFrame(preds)


def backtest_last_days(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Compare one-step-ahead predictions on history."""
    bundle = load_gb_forecaster()
    model = bundle["model"]
    feature_cols = bundle["features"]
    d = prepare_forecast_base(df).dropna()
    if d.empty:
        return pd.DataFrame()
    cutoff = d.index.max() - pd.Timedelta(days=days)
    sub = d[d.index >= cutoff]
    X = sub[feature_cols]
    y = sub["y_next"]
    pred = model.predict(X)
    out = sub[["net_power_mw"]].copy()
    out["y_next_actual"] = y
    out["y_next_pred"] = pred
    out["pct_error"] = (out["y_next_pred"] - out["y_next_actual"]).abs() / out["y_next_actual"].replace(0, np.nan) * 100
    return out
