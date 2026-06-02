"""
Generate synthetic hourly CPP dataset for calendar year 2025 (8760 rows).
Run: python data/generate_dataset.py from zeroshift directory.
"""
import os
import random

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT_PATH = os.path.join(ROOT, "data", "zeroshift_cppl.csv")

RNG = np.random.default_rng(42)
random.seed(42)

ANOMALY_TYPES = [
    "Steam Pressure Spike",
    "Bed Temperature Drop",
    "Drum Level Deviation",
    "Coal Feed Surge",
    "Turbine RPM Fluctuation",
    "ESP Inlet Temp Low",
    "Lube Oil Pressure Drop",
    "Power Generation Dip",
    "Auxiliary Power Surge",
    "Flue Gas Temp Anomaly",
]


def hour_shift(h: int) -> str:
    if 6 <= h < 14:
        return "Morning"
    if 14 <= h < 22:
        return "Afternoon"
    return "Night"


def inject_anomaly_row(row: pd.Series, severity: str, atype: str) -> pd.Series:
    """Apply drift to relevant columns based on anomaly type."""
    r = row.copy()
    drift_lo, drift_hi = 0.08, 0.15
    if severity == "ALERT":
        drift_lo, drift_hi = 0.15, 0.25
    elif severity == "CRITICAL":
        drift_lo, drift_hi = 0.26, 0.40

    d = float(RNG.uniform(drift_lo, drift_hi))
    sign = RNG.choice([-1.0, 1.0])

    if atype == "Steam Pressure Spike":
        r["steam_pressure_kgcm2"] *= 1 + d
    elif atype == "Bed Temperature Drop":
        r["bed_temp_celsius"] *= 1 - d
    elif atype == "Drum Level Deviation":
        r["drum_level_pct"] *= 1 + sign * d
    elif atype == "Coal Feed Surge":
        r["coal_feed_tph"] *= 1 + d
    elif atype == "Turbine RPM Fluctuation":
        r["turbine_rpm"] *= 1 + sign * d * 0.5
    elif atype == "ESP Inlet Temp Low":
        r["flue_gas_temp_celsius"] *= 1 - d
    elif atype == "Lube Oil Pressure Drop":
        r["lube_oil_pressure"] *= 1 - d
    elif atype == "Power Generation Dip":
        r["gross_power_mw"] *= 1 - d
    elif atype == "Auxiliary Power Surge":
        r["auxiliary_power_mw"] *= 1 + d
    elif atype == "Flue Gas Temp Anomaly":
        r["flue_gas_temp_celsius"] *= 1 + abs(sign) * d

    # Correlated multivariate stress so IF / autoencoder see separable outliers
    stress = {"WARNING": 0.15, "ALERT": 0.23, "CRITICAL": 0.35}[severity]
    r["gross_power_mw"] = float(r["gross_power_mw"] * (1 + sign * stress * 0.60))
    r["coal_feed_tph"] = float(r["coal_feed_tph"] * (1 + stress * 0.40))
    r["steam_pressure_kgcm2"] = float(
        np.clip(r["steam_pressure_kgcm2"] * (1 + stress * 0.35), 78.0, 95.0)
    )
    r["bed_temp_celsius"] = float(
        np.clip(r["bed_temp_celsius"] * (1 - sign * stress * 0.25), 760.0, 880.0)
    )
    r["turbine_rpm"] = float(np.clip(r["turbine_rpm"] + sign * stress * 12.0, 1475.0, 1525.0))
    r["flue_gas_temp_celsius"] = float(
        np.clip(r["flue_gas_temp_celsius"] * (1 - sign * stress * 0.12), 120.0, 175.0)
    )

    gross = float(r["gross_power_mw"])
    coal = float(r["coal_feed_tph"])
    aux_pct = float(row["auxiliary_power_mw"] / max(float(row["gross_power_mw"]), 1e-6))
    if atype != "Auxiliary Power Surge":
        aux = float(np.clip(gross * aux_pct, gross * 0.10, gross * 0.12))
        r["auxiliary_power_mw"] = aux
    else:
        r["auxiliary_power_mw"] = float(np.clip(float(r["auxiliary_power_mw"]), gross * 0.10, gross * 0.14))
    aux = float(r["auxiliary_power_mw"])
    r["net_power_mw"] = gross - aux
    r["plant_load_factor"] = (gross / 35.0) * 100.0
    r["coal_efficiency"] = r["net_power_mw"] / max(coal, 1e-6)
    r["co2_tons_per_hour"] = coal * 2.54

    r["anomaly_flag"] = 1
    return r


def select_anomaly_positions(n_rows: int, count: int = 200, min_gap: int = 5, margin: int = 48) -> list[int]:
    """Pick exactly `count` row indices with at least `min_gap` hours between them."""
    lo, hi = margin, n_rows - margin - 1
    if hi <= lo:
        raise ValueError("Dataset too small for anomaly injection")
    # Evenly spaced anchors (~43h apart on 8760 rows) — always satisfies min_gap >= 5
    anchors = np.linspace(lo, hi, count, dtype=int)
    chosen: list[int] = []
    for pos in anchors:
        pos = int(pos)
        if not chosen:
            chosen.append(pos)
            continue
        if pos - chosen[-1] < min_gap:
            pos = chosen[-1] + min_gap
        if pos > hi:
            break
        chosen.append(pos)
    # If linspace + gap enforcement trimmed the list, fill remaining slots from free ranges
    while len(chosen) < count:
        start = (chosen[-1] + min_gap) if chosen else lo
        if start > hi:
            break
        remaining = count - len(chosen)
        tail = np.linspace(start, hi, remaining, dtype=int)
        for pos in tail:
            pos = int(pos)
            if pos - chosen[-1] >= min_gap:
                chosen.append(pos)
            elif chosen[-1] + min_gap <= hi:
                chosen.append(chosen[-1] + min_gap)
        break
    if len(chosen) < count:
        raise RuntimeError(f"Could only place {len(chosen)} anomalies (need {count})")
    return chosen[:count]


def build_failure_in_48h(anomaly_positions: list[int], n_rows: int, lookback: int = 48) -> np.ndarray:
    """Label 1 for every hour in the 48h window immediately before each anomaly hour."""
    y = np.zeros(n_rows, dtype=int)
    for pos in anomaly_positions:
        start = max(0, pos - lookback)
        y[start:pos] = 1
    return y


def main() -> None:
    start = pd.Timestamp("2025-01-01 00:00:00")
    idx = pd.date_range(start, periods=8760, freq="h")

    rows = []
    for ts in idx:
        h = ts.hour
        dow = ts.dayofweek  # Mon=0
        month = ts.month

        shift = hour_shift(h)

        base = 31.5
        if shift == "Morning":
            base += 1.8
        elif shift == "Afternoon":
            base += 0.9
        else:
            base -= 1.4

        # Sunday maintenance window effect (lower MW on Sundays)
        if dow == 6:
            gross = float(RNG.uniform(18.0, 22.0))
        else:
            gross = base + float(RNG.normal(0, 0.35))
            gross += float(RNG.uniform(-1.2, 1.2))

        if month in (4, 5, 6):
            gross -= 0.5

        gross = float(np.clip(gross, 15.0, 36.0))

        coal = 13.8 + (gross - 31.5) * 0.12 + float(RNG.normal(0, 0.25))
        coal = float(np.clip(coal, 10.0, 20.0))

        steam_p = float(np.clip(87.0 + float(RNG.normal(0, 0.45)), 78.0, 95.0))
        steam_t = float(np.clip(515.0 + float(RNG.normal(0, 1.0)), 500.0, 530.0))
        bed = float(np.clip(820.0 + float(RNG.normal(0, 5.0)), 780.0, 880.0))
        flue = float(np.clip(148.0 + float(RNG.normal(0, 2.5)), 125.0, 170.0))
        drum = float(np.clip(50.0 + float(RNG.normal(0, 1.2)), 35.0, 65.0))
        rpm = float(np.clip(1500.0 + float(RNG.normal(0, 1.2)), 1485.0, 1515.0))
        lube = float(np.clip(3.0 + float(RNG.normal(0, 0.04)), 2.2, 3.5))

        aux_pct = float(RNG.uniform(0.10, 0.12))
        aux = gross * aux_pct
        net = gross - aux
        plf = (gross / 35.0) * 100.0
        coal_eff = net / max(coal, 1e-6)
        co2 = coal * 2.54

        # Ambient: seasonal + daily cycle
        seasonal = 18 + (month - 1) / 11.0 * 14
        if month in (4, 5, 6):
            seasonal = 38 + float(RNG.uniform(0, 4))
        ambient = seasonal + 4 * np.sin(2 * np.pi * h / 24) + float(RNG.normal(0, 0.8))
        ambient = float(np.clip(ambient, 12.0, 45.0))

        row = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "shift": shift,
            "gross_power_mw": gross,
            "coal_feed_tph": coal,
            "steam_pressure_kgcm2": steam_p,
            "steam_temp_celsius": steam_t,
            "bed_temp_celsius": bed,
            "flue_gas_temp_celsius": flue,
            "drum_level_pct": drum,
            "turbine_rpm": rpm,
            "lube_oil_pressure": lube,
            "auxiliary_power_mw": aux,
            "net_power_mw": net,
            "plant_load_factor": plf,
            "coal_efficiency": coal_eff,
            "co2_tons_per_hour": co2,
            "ambient_temp_celsius": ambient,
            "anomaly_flag": 0,
            "failure_in_48h": 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    n = len(df)

    chosen = select_anomaly_positions(n, count=200, min_gap=5, margin=48)
    severities = (["WARNING"] * 100) + (["ALERT"] * 65) + (["CRITICAL"] * 35)
    RNG.shuffle(severities)

    for i, pos in enumerate(chosen):
        atype = str(RNG.choice(ANOMALY_TYPES))
        sev = severities[i]
        updated = inject_anomaly_row(df.iloc[pos], sev, atype)
        for col in df.columns:
            if col in updated.index:
                df.at[pos, col] = updated[col]

    df["failure_in_48h"] = build_failure_in_48h(chosen, n, lookback=48)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(f"Anomalies injected: {int(df['anomaly_flag'].sum())}")
    print(f"failure_in_48h positives: {int(df['failure_in_48h'].sum())}")


if __name__ == "__main__":
    main()
