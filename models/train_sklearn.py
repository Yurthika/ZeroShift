"""Train sklearn models only (no TensorFlow). Run: python models/train_sklearn.py"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_score, r2_score
from sklearn.model_selection import train_test_split

from utils.data_loader import (
    add_engineered_features,
    lag_features_for_forecast,
    label_failure_in_48h,
    load_raw_dataframe,
    maintenance_feature_matrix,
    train_feature_matrix,
)
FORECAST_FEATURES = [
    "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_24h",
    "rolling_mean_6h", "rolling_std_6h", "coal_feed_tph",
    "steam_pressure_kgcm2", "steam_temp_celsius", "bed_temp_celsius",
    "hour_of_day", "day_of_week", "is_weekend", "shift_encoded", "ambient_temp_celsius",
]

os.makedirs(os.path.join(_ROOT, "models"), exist_ok=True)
df = add_engineered_features(load_raw_dataframe())

X_if, _ = train_feature_matrix(df)
y = df["anomaly_flag"].values
iso = IsolationForest(n_estimators=200, contamination=float(y.sum()) / len(y), random_state=42)
iso.fit(X_if.values)
pred = (iso.predict(X_if.values) == -1).astype(int)
print("Isolation Forest precision:", round(precision_score(y, pred) * 100, 1), "%")
joblib.dump(iso, os.path.join(_ROOT, "models", "isolation_forest.pkl"))

dlag = lag_features_for_forecast(df).dropna()
X = dlag[FORECAST_FEATURES].values
yt = dlag["y_next"].values
split = int(len(dlag) * 0.8)
gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
gbr.fit(X[:split], yt[:split])
pred_g = gbr.predict(X[split:])
print("GB R2:", round(r2_score(yt[split:], pred_g), 4))
joblib.dump({"model": gbr, "features": FORECAST_FEATURES}, os.path.join(_ROOT, "models", "gb_forecaster.pkl"))

dm = maintenance_feature_matrix(df)
feat_rf = [
    "rpm_dev", "lube_roll_mean", "lube_trend", "press_ratio", "coal_eff_roll",
    "power_roll_slope", "bed_roll_std", "coal_efficiency", "rolling_mean_power_6h", "rolling_std_power_6h",
]
Xr = dm[feat_rf].fillna(0).values
yr = label_failure_in_48h(df).values
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf.fit(Xtr, ytr)
print("RF test accuracy:", round((rf.predict(Xte) == yte).mean() * 100, 1), "%")
joblib.dump({"model": rf, "features": feat_rf}, os.path.join(_ROOT, "models", "rf_maintenance.pkl"))
print("Saved sklearn models.")
