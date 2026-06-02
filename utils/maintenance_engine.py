"""Predictive maintenance — RandomForest failure probability."""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import add_engineered_features, maintenance_feature_matrix
from utils.paths import project_root


EQUIPMENT = [
    {
        "name": "CFBC Boiler",
        "spec": "Thyssenkrupp Industries | 170 TPH",
        "bias": 0.00,
    },
    {
        "name": "Generator",
        "spec": "Siemens/TDPS | 35 MW | 11 KV",
        "bias": 0.02,
    },
    {
        "name": "Steam Turbine",
        "spec": "Triple Bleed Condensing | 1500 RPM",
        "bias": 0.03,
    },
    {"name": "Gear Box", "spec": "FLENDER | 6800→1500 RPM", "bias": 0.04},
    {
        "name": "ID Fan-1",
        "spec": "TLT Engg | 510 kW | 960 RPM | ABB VVFD",
        "bias": 0.025,
    },
    {
        "name": "PA Fan-1",
        "spec": "TLT Engg | 530 kW | 1460 RPM | ABB VVFD",
        "bias": 0.02,
    },
    {
        "name": "Boiler Feed Pump",
        "spec": "KSB Ltd | 650 kW | 6.6 KV | 2976 RPM",
        "bias": 0.015,
    },
    {"name": "ESP", "spec": "M/S Alstom | —", "bias": 0.035},
    {
        "name": "Ash Handling",
        "spec": "Macawber Beekay Dense Phase",
        "bias": 0.018,
    },
]


@st.cache_resource(show_spinner=False)
def load_rf_maintenance():
    path = os.path.join(project_root(), "models", "rf_maintenance.pkl")
    return joblib.load(path)


def build_maintenance_X(df: pd.DataFrame) -> pd.DataFrame:
    d0 = add_engineered_features(df)
    d = maintenance_feature_matrix(d0)
    feat_cols = [
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
    return d[feat_cols].copy().fillna(0)


def latest_failure_probability(df: pd.DataFrame) -> float:
    bundle = load_rf_maintenance()
    model = bundle["model"]
    cols = bundle["features"]
    X = build_maintenance_X(df).iloc[[-1]][cols]
    prob = float(model.predict_proba(X)[0, 1])
    return prob * 100.0


def equipment_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    base = latest_failure_probability(df)
    rows = []
    rng = np.random.default_rng(42)
    for i, eq in enumerate(EQUIPMENT):
        jitter = float(rng.normal(0, 3.0))
        p = float(np.clip(base + eq["bias"] * 100 + jitter, 0.5, 99.0))
        health = 100.0 - p
        hours_sm = int(2000 + rng.integers(0, 4000) - i * 120)
        days_next = int(np.clip(120 - p, 5, 180))
        if p < 30:
            risk = "LOW"
        elif p < 60:
            risk = "MEDIUM"
        elif p < 80:
            risk = "HIGH"
        else:
            risk = "CRITICAL"
        rows.append(
            {
                **eq,
                "failure_pct": p,
                "health_pct": health,
                "operating_hours": hours_sm,
                "days_to_maint": days_next,
                "risk": risk,
            }
        )
    return pd.DataFrame(rows)
