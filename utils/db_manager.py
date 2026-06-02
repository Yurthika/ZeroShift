"""SQLite helpers for ZeroShift."""
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.paths import project_root


def db_path() -> str:
    return os.path.join(project_root(), "database", "zeroshift.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path()), exist_ok=True)
    return sqlite3.connect(db_path(), check_same_thread=False)


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            anomaly_type TEXT,
            parameter TEXT,
            measured_value REAL,
            expected_min REAL,
            expected_max REAL,
            deviation_pct REAL,
            severity TEXT,
            detection_layer TEXT,
            status TEXT,
            assigned_to TEXT,
            recommendation TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            issue TEXT,
            root_cause TEXT,
            action_steps TEXT,
            equipment TEXT,
            savings_inr REAL,
            savings_kwh REAL,
            priority TEXT,
            created_at TEXT,
            resolved_at TEXT,
            resolved_by TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            report_type TEXT,
            period_start TEXT,
            period_end TEXT,
            created_at TEXT,
            file_size_kb REAL,
            sections_included TEXT
        )
        """
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM anomaly_log")
    if cur.fetchone()[0] == 0:
        seed_anomalies_and_recommendations(conn)
    conn.close()


def seed_anomalies_and_recommendations(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    anomalies = [
        (
            "2025-05-18 14:00:00",
            "Steam Pressure Spike",
            "steam_pressure_kgcm2",
            92.4,
            84.0,
            90.0,
            12.5,
            "WARNING",
            "Layer1",
            "NEW",
            "Unit Head",
            "Check SH-3 bypass and Wood Ward 505 tuning; reduce coal ~8%.",
            now,
        ),
        (
            "2025-05-18 09:30:00",
            "Bed Temperature Drop",
            "bed_temp_celsius",
            792.0,
            800.0,
            850.0,
            -18.0,
            "ALERT",
            "Both",
            "INVESTIGATING",
            "CPP Engineer",
            "Verify PA fan, siphon seals, and coal feeder stability.",
            now,
        ),
        (
            "2025-05-17 22:10:00",
            "ESP Inlet Temp Low",
            "flue_gas_temp_celsius",
            136.0,
            140.0,
            155.0,
            -11.0,
            "CRITICAL",
            "Layer2",
            "NEW",
            "Electrical",
            "Do not charge ESP; inspect APH fouling and air leakage.",
            now,
        ),
        (
            "2025-05-17 18:45:00",
            "Turbine RPM Fluctuation",
            "turbine_rpm",
            1488.0,
            1494.0,
            1506.0,
            8.0,
            "WARNING",
            "Layer1",
            "RESOLVED",
            "Mechanical",
            "Governor Wood Ward 505 trim; confirm lube oil 3.0 kgf/cm².",
            now,
        ),
        (
            "2025-05-17 06:00:00",
            "Drum Level Deviation",
            "drum_level_pct",
            38.0,
            44.0,
            56.0,
            -22.0,
            "ALERT",
            "Layer1",
            "NEW",
            "I&C",
            "Check BFP discharge pressure and feed control valve.",
            now,
        ),
        (
            "2025-05-16 21:20:00",
            "Lube Oil Pressure Drop",
            "lube_oil_pressure",
            2.35,
            2.5,
            3.2,
            -15.0,
            "CRITICAL",
            "Both",
            "INVESTIGATING",
            "Mechanical",
            "Start AOP immediately; verify overhead tank level.",
            now,
        ),
        (
            "2025-05-16 12:00:00",
            "Auxiliary Power Surge",
            "auxiliary_power_mw",
            4.1,
            3.0,
            3.8,
            18.0,
            "WARNING",
            "Layer1",
            "NEW",
            "Electrical",
            "Review VFD set points; optimize ID fan for draft control.",
            now,
        ),
        (
            "2025-05-15 19:30:00",
            "Coal Feed Surge",
            "coal_feed_tph",
            16.8,
            12.0,
            16.0,
            14.0,
            "ALERT",
            "Layer1",
            "RESOLVED",
            "Process",
            "Check bunker level, screw feeder, and belt speed.",
            now,
        ),
        (
            "2025-05-14 11:00:00",
            "Power Generation Dip",
            "gross_power_mw",
            26.5,
            29.0,
            34.0,
            -10.0,
            "WARNING",
            "Layer1",
            "NEW",
            "CPP Engineer",
            "Review coal quality and PA airflow distribution.",
            now,
        ),
        (
            "2025-05-13 03:00:00",
            "Flue Gas Temp Anomaly",
            "flue_gas_temp_celsius",
            160.0,
            140.0,
            155.0,
            12.0,
            "ALERT",
            "Layer2",
            "NEW",
            "Process",
            "Check ESP inlet and economizer ash bridging.",
            now,
        ),
        (
            "2025-05-12 16:20:00",
            "Steam Pressure Spike",
            "steam_pressure_kgcm2",
            91.2,
            84.0,
            90.0,
            9.0,
            "WARNING",
            "Layer1",
            "INVESTIGATING",
            "Unit Head",
            "Transient firing — trim coal master and SH bypass per SOP.",
            now,
        ),
        (
            "2025-05-11 20:10:00",
            "Drum Level Deviation",
            "drum_level_pct",
            62.0,
            44.0,
            56.0,
            15.0,
            "ALERT",
            "Layer1",
            "NEW",
            "I&C",
            "Verify three-element control and feedwater valve stroke.",
            now,
        ),
        (
            "2025-05-10 08:45:00",
            "ESP Inlet Temp Low",
            "flue_gas_temp_celsius",
            137.0,
            140.0,
            155.0,
            -8.0,
            "WARNING",
            "Layer2",
            "NEW",
            "Electrical",
            "APH soot blow sequence; hold ESP charging until >140°C.",
            now,
        ),
        (
            "2025-05-09 13:30:00",
            "Turbine RPM Fluctuation",
            "turbine_rpm",
            1496.0,
            1494.0,
            1506.0,
            5.0,
            "WARNING",
            "Layer1",
            "RESOLVED",
            "Mechanical",
            "Minor Wood Ward 505 retune completed.",
            now,
        ),
        (
            "2025-05-08 19:00:00",
            "Auxiliary Power Surge",
            "auxiliary_power_mw",
            3.9,
            3.0,
            3.8,
            9.0,
            "WARNING",
            "Layer1",
            "NEW",
            "Electrical",
            "Check VFD set points on ID fan and ESP hopper heaters.",
            now,
        ),
    ]
    cur.executemany(
        """
        INSERT INTO anomaly_log (
            timestamp, anomaly_type, parameter, measured_value,
            expected_min, expected_max, deviation_pct, severity,
            detection_layer, status, assigned_to, recommendation, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        anomalies,
    )

    recs = [
        (
            "Steam Pressure Spike — CPP1",
            "Steam pressure exceeded 90 kgf/cm² during afternoon peak.",
            "Transient coal firing plus SH-3 bypass margin low.",
            json.dumps(
                [
                    "Open SH-3 bypass per SOP",
                    "Trim Wood Ward 505 governor",
                    "Reduce coal feed ~8% for 20 minutes",
                ]
            ),
            "Boiler / Steam",
            28500.0,
            4200.0,
            "IMMEDIATE",
            now,
            None,
            None,
        ),
        (
            "Bed Temperature Drop",
            "Bed temperature trending below 800°C with unstable fluidization.",
            "PA airflow distribution or siphon seal leak suspected.",
            json.dumps(
                [
                    "Inspect PA fan dampers",
                    "Check siphon seals for hot spots",
                    "Verify coal feeder master demand",
                ]
            ),
            "CFBC Boiler",
            22000.0,
            3100.0,
            "IMMEDIATE",
            now,
            None,
            None,
        ),
        (
            "ESP Inlet Temperature Low",
            "Flue gas <140°C — ESP charging risk.",
            "APH fouling or excess air ingress.",
            json.dumps(
                [
                    "Hold ESP charging",
                    "APH soot blow sequence",
                    "Leak test economizer hopper",
                ]
            ),
            "ESP",
            45000.0,
            5100.0,
            "IMMEDIATE",
            now,
            None,
            None,
        ),
        (
            "Turbine RPM Fluctuation",
            "RPM oscillations around 1500 nominal.",
            "Governor tuning / lube header transient.",
            json.dumps(
                [
                    "Wood Ward 505 parameter review",
                    "Confirm lube 3.0 kgf/cm²",
                    "Vibration snapshot at bearing 2",
                ]
            ),
            "Steam Turbine",
            12000.0,
            1800.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "Drum Level Deviation",
            "Drum level below normal control band.",
            "BFP discharge / FCV hunting.",
            json.dumps(
                [
                    "Check BFP discharge pressure",
                    "Stroke test feed control valve",
                    "Verify deaerator level cascade",
                ]
            ),
            "BFP / Feedwater",
            18000.0,
            2600.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "Lube Oil Pressure Drop",
            "Lube header pressure <2.5 kgf/cm².",
            "AOP not auto-started or tank low.",
            json.dumps(
                [
                    "Start AOP immediately",
                    "Verify overhead tank level",
                    "Inspect pump suction strainer",
                ]
            ),
            "Lube Oil",
            52000.0,
            6000.0,
            "IMMEDIATE",
            now,
            None,
            None,
        ),
        (
            "Auxiliary Power Surge",
            "Aux power >12% of gross for sustained period.",
            "VFD operating point drift on ID fan.",
            json.dumps(
                [
                    "Review VFD PID parameters",
                    "Optimize ID fan for furnace draft",
                    "Check ESP hopper heaters load",
                ]
            ),
            "Electrical / VFD",
            15000.0,
            2100.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "Coal Feed Surge",
            "Coal feed spikes above 16 TPH.",
            "Bunker hang-up releasing slug.",
            json.dumps(
                [
                    "Inspect bunker level trends",
                    "Calibrate screw feeder speed",
                    "Check belt weigh feeder",
                ]
            ),
            "Fuel Handling",
            9000.0,
            1400.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "PA Fan VFD Optimization",
            "PA fan power margin high vs draft setpoint.",
            "Fan curve not aligned to bed inventory.",
            json.dumps(
                [
                    "PA VFD trim test",
                    "Bed inventory survey",
                    "Update DCS bias limits",
                ]
            ),
            "PA Fan-1",
            8000.0,
            1200.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "ID Fan Draft Hunting",
            "Furnace draft oscillations ±40 Pa.",
            "ID VFD PID gains aggressive.",
            json.dumps(
                [
                    "Soft-tune ID PID",
                    "Verify flue gas O2 cascade",
                    "Check expansion joint leaks",
                ]
            ),
            "ID Fan-1",
            11000.0,
            1600.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "Generator Winding Temperature",
            "Stator winding temperature marginal high.",
            "Hydrogen cooler performance seasonal.",
            json.dumps(
                [
                    "Increase H2 purity check",
                    "Clean cooler fins",
                    "Review MVAR setpoint",
                ]
            ),
            "Generator",
            6000.0,
            900.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "Ash Handling Dense Phase",
            "Dense phase pressure spikes during night shift.",
            "Air compressor unloading valve sticky.",
            json.dumps(
                [
                    "Service unloading valve",
                    "Verify ash silo level interlocks",
                    "Check booster air setpoint",
                ]
            ),
            "Ash Handling",
            7000.0,
            950.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "Gearbox Vibration Proxy",
            "RPM deviation correlated with torsional oscillation risk.",
            "Coupling alignment seasonal drift.",
            json.dumps(
                [
                    "Hot alignment check window",
                    "Inspect coupling bolts torque",
                    "Lube oil particle count",
                ]
            ),
            "Gear Box",
            13000.0,
            1500.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "ESP Hopper Ash Level",
            "Hopper level high alarms intermittent.",
            "Dense phase conveying air margin low.",
            json.dumps(
                [
                    "Manual hopper discharge test",
                    "Increase conveying air 5%",
                    "Inspect fluidizing pads",
                ]
            ),
            "ESP",
            9500.0,
            1300.0,
            "MONITOR",
            now,
            None,
            None,
        ),
        (
            "Boiler Feed Pump Recirc",
            "BFP recirc valve hunting at low load.",
            "Control valve actuator hysteresis.",
            json.dumps(
                [
                    "Stroke BFP recirc valve",
                    "Review minimum flow map",
                    "Check discharge pressure transmitter",
                ]
            ),
            "BFP",
            8500.0,
            1100.0,
            "MONITOR",
            now,
            None,
            None,
        ),
    ]
    cur.executemany(
        """
        INSERT INTO recommendations (
            title, issue, root_cause, action_steps, equipment,
            savings_inr, savings_kwh, priority, created_at, resolved_at, resolved_by
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        recs,
    )
    conn.commit()


def fetch_anomalies(
    limit: Optional[int] = None,
    severity: Optional[List[str]] = None,
    atype: Optional[List[str]] = None,
    status: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    q = "SELECT * FROM anomaly_log WHERE 1=1"
    params: List[Any] = []
    if severity:
        q += " AND severity IN (" + ",".join(["?"] * len(severity)) + ")"
        params.extend(severity)
    if atype:
        q += " AND anomaly_type IN (" + ",".join(["?"] * len(atype)) + ")"
        params.extend(atype)
    if status:
        q += " AND status IN (" + ",".join(["?"] * len(status)) + ")"
        params.extend(status)
    q += " ORDER BY datetime(timestamp) DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    cur = conn.cursor()
    cur.execute(q, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def count_new_anomalies() -> int:
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM anomaly_log WHERE status='NEW'")
    n = int(cur.fetchone()[0])
    conn.close()
    return n


def update_anomaly_status(aid: int, status: str) -> None:
    init_db()
    conn = get_connection()
    conn.execute("UPDATE anomaly_log SET status=? WHERE id=?", (status, aid))
    conn.commit()
    conn.close()


def fetch_recommendations(priority: Optional[str] = None) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    if priority:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM recommendations WHERE priority=? ORDER BY id DESC",
            (priority,),
        )
    else:
        cur = conn.cursor()
        cur.execute("SELECT * FROM recommendations ORDER BY id DESC")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def update_recommendation_priority(rid: int, priority: str, resolved_by: Optional[str] = None) -> None:
    init_db()
    conn = get_connection()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if priority == "DONE":
        conn.execute(
            "UPDATE recommendations SET priority=?, resolved_at=?, resolved_by=? WHERE id=?",
            (priority, now, resolved_by or "Operator", rid),
        )
    else:
        conn.execute("UPDATE recommendations SET priority=? WHERE id=?", (priority, rid))
    conn.commit()
    conn.close()


def insert_report_record(
    name: str,
    report_type: str,
    period_start: str,
    period_end: str,
    file_size_kb: float,
    sections: List[str],
) -> None:
    init_db()
    conn = get_connection()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO reports (name, report_type, period_start, period_end, created_at, file_size_kb, sections_included)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            name,
            report_type,
            period_start,
            period_end,
            now,
            file_size_kb,
            json.dumps(sections),
        ),
    )
    conn.commit()
    conn.close()


def fetch_reports(limit: int = 10) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM reports ORDER BY datetime(created_at) DESC LIMIT ?", (limit,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def log_engine_anomalies(rows: List[Dict[str, Any]]) -> None:
    """Bulk insert synthetic detections (optional sync from engine)."""
    init_db()
    conn = get_connection()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for r in rows:
        conn.execute(
            """
            INSERT INTO anomaly_log (
                timestamp, anomaly_type, parameter, measured_value,
                expected_min, expected_max, deviation_pct, severity,
                detection_layer, status, assigned_to, recommendation, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                r.get("timestamp"),
                r.get("anomaly_type"),
                r.get("parameter"),
                r.get("measured_value"),
                r.get("expected_min"),
                r.get("expected_max"),
                r.get("deviation_pct"),
                r.get("severity"),
                r.get("detection_layer"),
                r.get("status", "NEW"),
                r.get("assigned_to", "CPP Team"),
                r.get("recommendation", ""),
                now,
            ),
        )
    conn.commit()
    conn.close()
