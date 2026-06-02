"""Load CSV, parse timestamps, feature engineering pipeline."""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.paths import project_root


def _csv_path() -> str:
    return os.path.join(project_root(), "data", "zeroshift_cppl.csv")


def load_raw_dataframe() -> pd.DataFrame:
    """Load CSV without Streamlit caching (for training scripts)."""
    path = _csv_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python data/generate_dataset.py"
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


@st.cache_data(show_spinner=False)
def load_raw_csv() -> pd.DataFrame:
    return load_raw_dataframe()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features not present in raw CSV."""
    out = df.copy()
    out["rolling_mean_power_6h"] = out["gross_power_mw"].rolling(6, min_periods=1).mean()
    out["rolling_std_power_6h"] = out["gross_power_mw"].rolling(6, min_periods=1).std().fillna(0)
    out["power_change_rate"] = out["gross_power_mw"].diff().fillna(0)
    out["coal_per_mw"] = out["coal_feed_tph"] / out["net_power_mw"].replace(0, np.nan)
    out["coal_per_mw"] = out["coal_per_mw"].fillna(out["coal_per_mw"].median())
    out["pressure_temp_ratio"] = out["steam_pressure_kgcm2"] / out["steam_temp_celsius"].replace(0, np.nan)
    out["hour_of_day"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["is_weekend"] = out["day_of_week"] >= 5

    # Shift one-hot for models
    shift_map = {"Morning": 0, "Afternoon": 1, "Night": 2}
    out["shift_encoded"] = out["shift"].map(shift_map).fillna(0).astype(int)
    return out


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    df = load_raw_csv()
    return add_engineered_features(df)


def train_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Features for Isolation Forest / autoencoder input columns."""
    cols = [
        "gross_power_mw",
        "coal_feed_tph",
        "steam_pressure_kgcm2",
        "steam_temp_celsius",
        "bed_temp_celsius",
        "turbine_rpm",
        "lube_oil_pressure",
        "flue_gas_temp_celsius",
        "drum_level_pct",
        "auxiliary_power_mw",
    ]
    return df[cols].copy(), cols


def lag_features_for_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag and rolling features for gradient boosting."""
    d = df.copy()
    target = d["net_power_mw"]
    d["lag_1h"] = target.shift(1)
    d["lag_2h"] = target.shift(2)
    d["lag_3h"] = target.shift(3)
    d["lag_6h"] = target.shift(6)
    d["lag_24h"] = target.shift(24)
    d["rolling_mean_6h"] = target.rolling(6, min_periods=1).mean()
    d["rolling_std_6h"] = target.rolling(6, min_periods=1).std().fillna(0)
    d["y_next"] = target.shift(-1)
    return d


def maintenance_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Features for RF maintenance classifier."""
    d = df.copy()
    d["rpm_dev"] = (d["turbine_rpm"] - 1500.0).abs()
    d["lube_roll_mean"] = d["lube_oil_pressure"].rolling(12, min_periods=1).mean()
    d["lube_trend"] = d["lube_oil_pressure"].diff(6).fillna(0)
    d["press_ratio"] = d["pressure_temp_ratio"]
    d["coal_eff_roll"] = d["coal_efficiency"].rolling(24, min_periods=1).mean()
    d["power_roll_slope"] = d["gross_power_mw"].diff(24).fillna(0)
    d["bed_roll_std"] = d["bed_temp_celsius"].rolling(24, min_periods=1).std().fillna(0)
    return d


def label_failure_in_48h(df: pd.DataFrame) -> pd.Series:
    """1 for each hour in the 48h window before an anomaly (precursor label)."""
    if "failure_in_48h" in df.columns:
        return df["failure_in_48h"].astype(int)
    flags = df["anomaly_flag"].values
    n = len(flags)
    y = np.zeros(n, dtype=int)
    anomaly_idx = np.flatnonzero(flags == 1)
    for pos in anomaly_idx:
        start = max(0, int(pos) - 48)
        y[start:pos] = 1
    return pd.Series(y, index=df.index)
