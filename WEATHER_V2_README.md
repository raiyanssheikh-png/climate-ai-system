# 🌦️ Climate AI — Extreme Weather Prediction System (v2)

An industry-level machine-learning system that predicts extreme weather events across **20 global cities** and produces a **seasonal disaster-risk forecast**. Built as a Final Year Project at Sindh Madressatul Islam University (SMIU), Karachi.

**Authors:** Syed Bilal · Raiyan Sheikh · Numra Amjad

---

## 🎯 What This Project Does

- Collects 11+ years of hourly weather data (2015–2026) for 20 cities across 6 continents from the Open-Meteo API
- Engineers 54 features in 5 groups (temporal, lag, rolling, fusion, geographic)
- Benchmarks **11 machine-learning models** (5 regression + 6 classification)
- Selects the best model scientifically using cross-validation and computational-efficiency analysis
- Explains predictions with **5 XAI methods**: SHAP, LIME, Permutation Feature Importance, Integrated Gradients, and GradCAM
- Produces an honest **seasonal risk forecast** for summer 2026 disaster planning

---

## 🏆 Key Results

| Task | Champion Model | Performance |
|------|---------------|-------------|
| Temperature regression | LightGBM | R² = 0.9965, RMSE = 0.65°C |
| Heatwave detection | LightGBM (tuned) | F1 = 0.9942, AUC = 1.000 |

---

## 🤖 Models Benchmarked

**Regression:** Ridge · Random Forest · XGBoost · LightGBM · CatBoost
**Classification:** Logistic Regression · Random Forest · XGBoost · LightGBM · CatBoost · 1D-CNN (deep learning)

---

## 🧠 Explainability (XAI)

| Method | Type | Purpose |
|--------|------|---------|
| SHAP | Global + local | Feature contributions via game theory |
| LIME | Local | Single-prediction explanations |
| PFI | Global | Permutation-based importance |
| Integrated Gradients | Local (neural) | Attribution for the CNN |
| GradCAM | Local (neural) | Activation mapping for the CNN |

---

## 🚀 Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Repository Structure

```
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── dashboard_data.csv.gz # Compact data for the dashboard
│   ├── *_results.json        # Model benchmark leaderboards
│   └── table*.csv            # Forecast + experiment tables
└── models/
    ├── reg_lightgbm.pkl       # Champion regression model
    ├── clf_heatwave_tuned.pkl # Champion heatwave model
    └── feature_list.json
```

---

## 📊 Data Source

Open-Meteo Historical Weather API (ERA5 reanalysis), 2015–2026, 20 cities, ~2 million hourly records.

---

*This system is built for research and disaster-preparedness planning. Seasonal forecasts are probabilistic risk estimates, not exact predictions.*
