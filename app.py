#!/usr/bin/env python3
"""
AI-Driven Extreme Weather Prediction System — v2
4 Tasks · 11 Models Benchmarked · 5 XAI Methods · Seasonal Forecast
Authors: Syed Bilal, Raiyan Sheikh & Numra Amjad — SMIU Karachi
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Climate AI — Extreme Weather Prediction",
                   page_icon="🌦️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:700; text-align:center; padding:.5rem 0; }
    .sub-header  { font-size:.95rem; text-align:center; opacity:.7; margin-bottom:1.2rem; }
    .alert-green  { border-left:6px solid #2e7d32; background:rgba(46,125,50,.12); padding:.8rem 1rem; border-radius:6px; font-weight:600; }
    .alert-yellow { border-left:6px solid #f9a825; background:rgba(249,168,37,.12); padding:.8rem 1rem; border-radius:6px; font-weight:600; }
    .alert-orange { border-left:6px solid #e65100; background:rgba(230,81,0,.12); padding:.8rem 1rem; border-radius:6px; font-weight:600; }
    .alert-red    { border-left:6px solid #c62828; background:rgba(198,40,40,.12); padding:.8rem 1rem; border-radius:6px; font-weight:600; }
    .sec { font-size:1.05rem; font-weight:600; margin:.8rem 0 .3rem 0; }
</style>
""", unsafe_allow_html=True)

C = {"blue":"#4472CA","red":"#E15759","green":"#59A14F","orange":"#F28E2B",
     "teal":"#4E9FA8","purple":"#B07AA1","gray":"#76787A"}
HW_CITIES = {"Karachi","Delhi","Mumbai","Dhaka"}
CLASS_NAMES = {0:"Normal",1:"Heatwave",2:"Heavy Rain",3:"Storm"}
CLASS_COLORS = {"Normal":C["green"],"Heatwave":C["orange"],"Heavy Rain":C["blue"],"Storm":C["red"]}

# ── Paths ──────────────────────────────────────────────────────
@st.cache_data
def find_dirs():
    for base in [Path.cwd(), Path("/mount/src/extreme-weather-prediction-v2"),
                 Path.home()/"weather_v2"/"deploy"]:
        if (base/"data").exists() and (base/"models").exists():
            return base/"data", base/"models"
    for p in Path.cwd().rglob("dashboard_data.csv.gz"):
        return p.parent, p.parent.parent/"models"
    return Path("data"), Path("models")

@st.cache_data
def load_data(data_dir):
    df = pd.read_csv(data_dir/"dashboard_data.csv.gz", low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["year"]  = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["hour"]  = df["datetime"].dt.hour
    return df

@st.cache_data
def load_table(data_dir, name):
    p = data_dir/name
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_json(data_dir, name):
    p = data_dir/name
    if p.exists():
        with open(p) as f: return json.load(f)
    return None

@st.cache_resource
def load_models(models_dir):
    m = {}
    for key, fn in [("temp","reg_lightgbm.pkl"),
                    ("heatwave","clf_heatwave_tuned.pkl"),
                    ("rain","champion_target_rain.pkl"),
                    ("disaster","champion_target_disaster.pkl")]:
        p = models_dir/fn
        if p.exists(): m[key] = joblib.load(p)
    fl = models_dir/"feature_list.json"
    if fl.exists():
        with open(fl) as f: m["features"] = json.load(f)
    return m


def main():
    data_dir, models_dir = find_dirs()
    df = load_data(data_dir)
    models = load_models(models_dir)
    feats = models.get("features", [])

    st.markdown('<div class="main-header">🌦️ Climate AI — Extreme Weather Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">20 Global Cities · 4 Prediction Tasks · 11 Models Benchmarked · 5 XAI Methods</div>', unsafe_allow_html=True)

    cities = sorted(df["city"].unique())
    with st.sidebar:
        st.markdown("### 🌍 Controls")
        sel = st.selectbox("City", cities, index=cities.index("Karachi") if "Karachi" in cities else 0)
        st.markdown("---")
        st.markdown("**4 Prediction Tasks**")
        st.markdown("🌡️ Temperature — R² 0.9965\n☀️ Heatwave — F1 0.9942\n🌧️ Rainfall — F1 1.000\n🚨 Disaster — F1 0.9993")
        st.markdown("---")
        st.markdown("**Champion:** LightGBM (all tasks)")
        st.markdown("---")
        st.markdown("**Team — SMIU**\nSyed Bilal · Raiyan Sheikh · Numra Amjad")

    city_df = df[df["city"]==sel].copy()
    threshold = 40.0 if sel in HW_CITIES else 35.0

    t1, t2, t3, t4 = st.tabs([
        "🔮 Live Predictions",
        "📊 Model Benchmark",
        "🔥 Seasonal Forecast",
        "📈 City Trends",
    ])

    # ── TAB 1: LIVE PREDICTIONS (all 4 tasks) ─────────────────
    with t1:
        st.subheader(f"🔮 Predictions for {sel}")
        if len(city_df)==0:
            st.warning("No data."); st.stop()

        latest = city_df.iloc[-1:]
        avail = [f for f in feats if f in city_df.columns]
        has_models = len(avail)==len(feats) and len(feats)>0

        cur_temp = latest["temperature_2m"].values[0]
        cur_hum  = latest["relative_humidity_2m"].values[0]
        cur_rain = latest["precipitation"].values[0]
        cur_wind = latest["windspeed_10m"].values[0]

        if has_models:
            X = latest[feats]
            hw_prob   = models["heatwave"].predict_proba(X)[0][1] if "heatwave" in models else 0
            rain_prob = models["rain"].predict_proba(X)[0][1] if "rain" in models else 0
            dis_pred  = int(models["disaster"].predict(X)[0]) if "disaster" in models else 0
            if hasattr(dis_pred, "__len__"): dis_pred = int(dis_pred[0])
        else:
            hw_prob = 1.0 if cur_temp>=threshold else 0.0
            rain_prob = 1.0 if cur_rain>1.0 else 0.0
            dis_pred = 1 if cur_temp>=threshold else 2 if cur_rain>1.0 else 0

        dis_name = CLASS_NAMES.get(dis_pred, "Normal")

        # Alert
        if dis_pred==3 or cur_wind>=40:
            atxt, acls = "🚨 STORM ALERT", "alert-red"
        elif hw_prob>0.7 or cur_temp>42:
            atxt, acls = "⚠️ HEATWAVE WARNING", "alert-orange"
        elif rain_prob>0.5 or cur_rain>1.0:
            atxt, acls = "⛈️ RAIN ADVISORY", "alert-yellow"
        else:
            atxt, acls = "✅ NORMAL CONDITIONS", "alert-green"
        st.markdown(f'<div class="{acls}">{atxt}</div>', unsafe_allow_html=True)
        st.markdown("")

        # 4 task cards
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🌡️ Temperature", f"{cur_temp:.1f} °C")
        c2.metric("☀️ Heatwave Risk", f"{hw_prob*100:.0f}%")
        c3.metric("🌧️ Rain Risk", f"{rain_prob*100:.0f}%")
        c4.metric("🚨 Disaster Class", dis_name)

        st.markdown("---")
        st.markdown('<p class="sec">🌡️ Temperature — Last 14 Days</p>', unsafe_allow_html=True)
        rec = city_df.tail(24*14)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec["datetime"], y=rec["temperature_2m"],
                                 mode="lines", name="Temperature", line=dict(color=C["red"],width=2)))
        fig.add_hline(y=threshold, line_dash="dash", line_color=C["orange"],
                      annotation_text=f"Heatwave threshold ({threshold}°C)")
        fig.update_layout(height=360, xaxis_title="Date", yaxis_title="°C", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="sec">☀️ Heatwave Risk</p>', unsafe_allow_html=True)
            g = go.Figure(go.Indicator(mode="gauge+number", value=hw_prob*100,
                gauge={"axis":{"range":[0,100]},"bar":{"color":C["orange"]},
                       "steps":[{"range":[0,30],"color":"rgba(89,161,79,.2)"},
                                {"range":[30,70],"color":"rgba(242,142,43,.2)"},
                                {"range":[70,100],"color":"rgba(225,87,89,.2)"}]},
                title={"text":"Heatwave %"}))
            g.update_layout(height=260, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(g, use_container_width=True, theme="streamlit")
        with col2:
            st.markdown('<p class="sec">🌧️ Rain Risk</p>', unsafe_allow_html=True)
            g2 = go.Figure(go.Indicator(mode="gauge+number", value=rain_prob*100,
                gauge={"axis":{"range":[0,100]},"bar":{"color":C["blue"]},
                       "steps":[{"range":[0,30],"color":"rgba(89,161,79,.2)"},
                                {"range":[30,70],"color":"rgba(78,159,168,.2)"},
                                {"range":[70,100],"color":"rgba(68,114,202,.3)"}]},
                title={"text":"Rain %"}))
            g2.update_layout(height=260, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(g2, use_container_width=True, theme="streamlit")

    # ── TAB 2: MODEL BENCHMARK (all 4 tasks) ──────────────────
    with t2:
        st.subheader("📊 Model Benchmark — All 4 Tasks")

        task_sum = load_json(data_dir, "task_summary.json")
        if task_sum:
            cols = st.columns(4)
            tasks = [("🌡️ Temperature","temperature"),("☀️ Heatwave","heatwave"),
                     ("🌧️ Rainfall","rainfall"),("🚨 Disaster","disaster")]
            for col,(label,key) in zip(cols,tasks):
                d = task_sum.get(key,{})
                col.metric(label, f"{d.get('value','?')}", f"{d.get('champion','')}")

        st.markdown('<p class="sec">🌡️ Regression — Temperature (5 models)</p>', unsafe_allow_html=True)
        reg = load_json(data_dir, "regression_results.json")
        if reg:
            rdf = pd.DataFrame(reg).sort_values("RMSE")
            fig = go.Figure(go.Bar(y=rdf["Model"], x=rdf["RMSE"], orientation="h",
                marker_color=C["blue"], text=rdf["RMSE"].round(3), textposition="outside"))
            fig.update_layout(height=300, xaxis_title="RMSE °C (lower=better)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown('<p class="sec">🌧️ Rainfall (5 models)</p>', unsafe_allow_html=True)
            rb = load_table(data_dir, "benchmark_target_rain.csv")
            if rb is not None:
                fig = go.Figure(go.Bar(y=rb["Model"], x=rb["F1"], orientation="h",
                    marker_color=C["teal"], text=rb["F1"].round(3), textposition="outside"))
                fig.update_layout(height=300, xaxis_title="F1", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        with cc2:
            st.markdown('<p class="sec">🚨 Disaster (5 models)</p>', unsafe_allow_html=True)
            db = load_table(data_dir, "benchmark_target_disaster.csv")
            if db is not None:
                fig = go.Figure(go.Bar(y=db["Model"], x=db["F1"], orientation="h",
                    marker_color=C["purple"], text=db["F1"].round(3), textposition="outside"))
                fig.update_layout(height=300, xaxis_title="Weighted F1", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        st.markdown('<p class="sec">☀️ Heatwave — Classification (6 models incl. CNN)</p>', unsafe_allow_html=True)
        clf = load_json(data_dir, "classification_results.json")
        if clf:
            cdf = pd.DataFrame(clf).sort_values("F1", ascending=False)
            fig = go.Figure(go.Bar(y=cdf["Model"], x=cdf["F1"], orientation="h",
                marker_color=C["orange"], text=cdf["F1"].round(3), textposition="outside"))
            fig.update_layout(height=340, xaxis_title="F1", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        st.markdown('<p class="sec">🔧 Staged Hyperparameter Tuning (Heatwave)</p>', unsafe_allow_html=True)
        hp = load_table(data_dir, "table9_hyperparameter_stages.csv")
        if hp is not None:
            st.dataframe(hp, use_container_width=True, hide_index=True)

    # ── TAB 3: SEASONAL FORECAST ──────────────────────────────
    with t3:
        st.subheader("🔥 Seasonal Risk Forecast — Summer 2026")
        st.info("Historical climatology + warming trend. Risk = % of recent-year hours above heatwave threshold. Honest probabilistic outlook for disaster planning.")

        fc = load_table(data_dir, "table15_seasonal_forecast.csv")
        if fc is not None:
            pivot = fc.pivot(index="City", columns="Month", values="Heatwave_Risk_%")
            for m in ["Jun","Jul","Aug"]:
                if m not in pivot.columns: pivot[m]=0
            pivot = pivot[["Jun","Jul","Aug"]]
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
            fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="YlOrRd",
                            labels={"color":"Risk %"}, aspect="auto")
            fig.update_layout(height=560, title="Heatwave Risk % by City and Month")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            city_fc = fc[fc["City"]==sel]
            if len(city_fc)>0:
                st.markdown(f'<p class="sec">🏙️ {sel} — Summer 2026 Detail</p>', unsafe_allow_html=True)
                st.dataframe(city_fc, use_container_width=True, hide_index=True)

    # ── TAB 4: CITY TRENDS ────────────────────────────────────
    with t4:
        st.subheader(f"📈 Climate Trends — {sel}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="sec">🌡️ Yearly Temperature</p>', unsafe_allow_html=True)
            yt = city_df.groupby("year")["temperature_2m"].mean().reset_index()
            fig = px.line(yt, x="year", y="temperature_2m", markers=True, color_discrete_sequence=[C["red"]])
            fig.update_layout(height=300, yaxis_title="Avg °C")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        with col2:
            st.markdown('<p class="sec">☀️ Heatwave Hours/Year</p>', unsafe_allow_html=True)
            city_df["hw"] = (city_df["temperature_2m"]>=threshold).astype(int)
            hwy = city_df.groupby("year")["hw"].sum().reset_index()
            fig = px.bar(hwy, x="year", y="hw", color="hw", color_continuous_scale="YlOrRd", text="hw")
            fig.update_layout(height=300, yaxis_title="Hours", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<p class="sec">🌧️ Monthly Rainfall</p>', unsafe_allow_html=True)
            mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            rm = city_df.groupby("month")["precipitation"].sum().reset_index()
            rm["m"] = rm["month"].apply(lambda x: mn[x-1])
            fig = px.bar(rm, x="m", y="precipitation", color="precipitation", color_continuous_scale="Blues")
            fig.update_layout(height=300, yaxis_title="mm", xaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        with col4:
            st.markdown('<p class="sec">🔬 SHAP — Heatwave Drivers</p>', unsafe_allow_html=True)
            shp = load_table(data_dir, "table10_shap_importance.csv")
            if shp is not None:
                col = "Mean_Abs_SHAP" if "Mean_Abs_SHAP" in shp.columns else shp.columns[-1]
                fcol = "Feature" if "Feature" in shp.columns else shp.columns[0]
                top = shp.head(8).sort_values(col)
                fig = px.bar(top, x=col, y=fcol, orientation="h", color=col, color_continuous_scale="Purples")
                fig.update_layout(height=300, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")


if __name__ == "__main__":
    main()
