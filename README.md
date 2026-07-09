# 🌦️ Climate AI — Multi-Horizon Extreme Weather Forecasting System

A machine-learning system that forecasts **temperature, heatwaves, heavy rainfall, and extreme-weather events** 30, 60, and 90 days ahead for 10 climate-diverse cities worldwide — with a live, interactive dashboard.

**Final Year Project** · Department of AI & Mathematical Sciences
**Sindh Madressatul Islam University (SMIU), Karachi**

**Team:** Raiyan Sheikh · Syed Bilal  **Supervisor:** Syed Azeem Inam

---

## 🚀 Live Dashboard

The interactive dashboard is deployed on Streamlit Cloud. It fetches live weather data and generates fresh forecasts on demand.

> Select any city to see its 30 / 60 / 90-day outlook across all four forecast targets, with interactive charts and an all-cities comparison.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 What It Does

The system forecasts four targets at three horizons (30 / 60 / 90 days):

| Target | Type | What it predicts |
|--------|------|------------------|
| 🌡️ Temperature | Regression | Daily mean temperature (°C) |
| 🔥 Heatwave | Classification | Probability of a heatwave |
| 🌧️ Heavy Rain | Classification | Probability of heavy rainfall (>10 mm) |
| ⚠️ Disaster | Classification | Probability of any extreme event |

**Champion model:** LightGBM (gradient boosting), selected after benchmarking six model families plus a Transformer.

---

## 🌍 Cities Covered

Ten cities spanning the full range of global climate zones, each with quality-controlled historical station data:

| City | Climate Zone | City | Climate Zone |
|------|-------------|------|-------------|
| Lagos | Tropical | Miami | Subtropical |
| Jakarta | Tropical | Phoenix | Arid |
| Delhi | Subtropical | Chicago | Continental |
| Madrid | Mediterranean | Moscow | Continental |
| Sydney | Temperate | Rotterdam | Maritime |

---

## 🧠 How It Works

1. **Data** — Daily weather (2015–2026) from the Open-Meteo Archive API (ERA5 reanalysis), validated against real GHCN-Daily station observations.
2. **Features** — 25 predictive features per forecast: rolling averages (7/30/90-day) of temperature, humidity, rainfall, wind and pressure, plus geographic and seasonal encodings.
3. **Models** — Separate LightGBM models per target and horizon, trained leakage-free (predictors use only past information; targets are set in the future).
4. **Forecasting** — Given recent weather up to today, the models project each target 30/60/90 days forward.
5. **Dashboard** — A Streamlit app fetches live data, runs the models, and visualizes the results.

---

## 📈 Performance (Honest Reporting)

Measured on held-out test data (cities seen during training):

| Target | Metric | Score (90-day) |
|--------|--------|----------------|
| Temperature | R² | 0.91 |
| Heatwave | F1 / AUC | 0.77 / 0.97 |
| Disaster | F1 / AUC | 0.57 / 0.85 |
| Rain | F1 / AUC | 0.32 / 0.78 |

**Note on rainfall:** Long-range rainfall is inherently the hardest target — this is reported transparently rather than hidden. Temperature and heatwave are the most reliable outputs.

The models were also benchmarked against a day-of-year **climatology baseline**: while climatology skill collapses with horizon (R² 0.81 → 0.08 at 90 days), the ML model maintains R² ≈ 0.91, demonstrating genuine dynamic forecast skill.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `app.py` | The Streamlit dashboard |
| `forecast_*.pkl` | Trained forecast models (4 targets × 3 horizons) |
| `daily_forecast.csv.gz` | City metadata + processed daily data |
| `feature_list.json` | Feature definitions |
| `requirements.txt` | Python dependencies |
| `*_results.csv`, `table*.csv` | Benchmarking, cross-validation & explainability results |

---

## 🛠️ Tech Stack

- **Python** · **LightGBM** · **scikit-learn** · **pandas** · **NumPy**
- **Streamlit** + **Plotly** (dashboard & visualizations)
- **Data:** Open-Meteo (ERA5 reanalysis), GHCN-Daily station observations

---

## 📄 Data Attribution

Weather data sourced from the [Open-Meteo Archive API](https://open-meteo.com/) (ERA5 reanalysis) and the [NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) station network.

---

## ⚠️ Disclaimer

This is an academic research and demonstration project. Forecasts are **not** official weather warnings and should not be used for operational or safety-critical decisions. For authoritative forecasts, consult your national meteorological service.

---

*© 2026 · Sindh Madressatul Islam University, Karachi*
