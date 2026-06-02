# ZeroShift

**ZeroShift** is an AI-powered energy optimization and anomaly detection cockpit for the captive power plant (CPP1) at **UltraTech Cement — Balaji Cement Works**. It ships with a full-year synthetic hourly dataset, pre-trained scikit-learn + TensorFlow models, SQLite-backed workflows, Plotly visualizations, GPT-3.5-turbo Q&A (optional), and PDF reporting.

## Quick start (local)

```bash
cd zeroshift
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python data/generate_dataset.py
python models/train_models.py
streamlit run app.py
```

> **Streamlit Cloud:** set the app root to the `zeroshift/` folder and main file to `app.py`. Do **not** auto-train models at startup—commit the generated `models/*.pkl`, `models/autoencoder_weights.h5`, and `models/autoencoder_scaler.pkl` after running `train_models.py` locally.

## Features

- **Overview / Energy / Analytics:** KPIs, synchronized Plotly charts, correlations, benchmarking.
- **Two-layer anomalies:** Isolation Forest + LSTM autoencoder (`utils/anomaly_engine.py`).
- **Forecasting:** Gradient Boosting forecaster with 24–168h iterative horizon (`utils/forecast_engine.py`).
- **Predictive maintenance:** RandomForest 48h risk horizon with equipment gauges (`utils/maintenance_engine.py`).
- **SQLite:** `anomaly_log`, `recommendations`, `reports` tables (`utils/db_manager.py`).
- **AI Agent:** OpenAI `gpt-3.5-turbo` with deterministic rule-based fallback (`utils/ai_agent.py`).
- **PDF reports:** ReportLab + Plotly/Kaleido chart embeds (`utils/report_generator.py`).

## Data

`data/generate_dataset.py` creates `data/zeroshift_cppl.csv` (8,760 hourly rows for 2025) with realistic CPP signals and ~200 injected anomalies.

## License / disclaimer

Synthetic demo data and heuristics are for illustration only—not a substitute for plant safety systems, OEM manuals, or site-specific engineering judgement.
