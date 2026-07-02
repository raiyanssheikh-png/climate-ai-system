#!/usr/bin/env python3
"""
Climate AI — Live Multi-Horizon Weather Forecasting Dashboard
Forecasts temperature, heatwave, rainfall & disaster 30/60/90 days ahead
for 10 climate-diverse cities, using trained LightGBM models + live data.

Authors: Raiyan Sheikh, Syed Bilal — Supervisor: Syed Azeem Inam (SMIU Karachi)

RUN:  streamlit run app_live.py
"""
import streamlit as st
import pandas as pd, numpy as np, json, joblib, requests, time
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Climate AI — Live Forecast",
                   page_icon="🌦️", layout="wide", initial_sidebar_state="expanded")

# ── Light theme ────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background:#fafafa; }
  .hdr { font-size:1.9rem; font-weight:700; text-align:center; color:#2c3e50; padding:.4rem 0; }
  .sub { font-size:.9rem; text-align:center; color:#7f8c8d; margin-bottom:1rem; }
  .exp { font-size:.85rem; color:#5d6d7e; background:#f0f3f7; border-left:4px solid #85a3c4;
         padding:.5rem .8rem; border-radius:4px; margin:.3rem 0 1rem 0; }
  .card { background:#fff; border:1px solid #e6e9 ed; border-radius:10px; padding:1rem;
          box-shadow:0 1px 3px rgba(0,0,0,.05); }
  .big { font-size:2.4rem; font-weight:700; color:#2c3e50; }
  .lbl { font-size:.8rem; color:#95a5a6; text-transform:uppercase; letter-spacing:.5px; }
  .alert-hot { color:#c0392b; font-weight:700; }
  .alert-ok  { color:#27ae60; font-weight:600; }
</style>
""", unsafe_allow_html=True)

CITIES = ["Lagos","Jakarta","Delhi","Sydney","Madrid",
          "Rotterdam","Miami","Chicago","Phoenix","Moscow"]
ZONE = {"Lagos":"Tropical","Jakarta":"Tropical","Delhi":"Subtropical","Miami":"Subtropical",
        "Phoenix":"Arid","Madrid":"Mediterranean","Sydney":"Temperate","Rotterdam":"Maritime",
        "Chicago":"Continental","Moscow":"Continental"}
FEATURES = ['latitude','longitude','coastal','climate_zone_id','continent_id','city_id',
            'temp_mean_roll7','temp_mean_roll30','temp_mean_roll90',
            'humidity_roll7','humidity_roll30','humidity_roll90',
            'precip_sum_roll7','precip_sum_roll30','precip_sum_roll90',
            'wind_max_roll7','wind_max_roll30','wind_max_roll90',
            'pressure_roll7','pressure_roll30','pressure_roll90',
            'doy_sin','doy_cos','month','year']

def exp(t): st.markdown(f'<div class="exp">💡 {t}</div>', unsafe_allow_html=True)

# ── Locate files (FLAT repo: everything sits next to app.py) ──
@st.cache_data
def find_base():
    here = Path(__file__).resolve().parent
    candidates = [here, Path.cwd(), Path.home()/"weather_v2"]
    for base in candidates:
        # flat layout: models + metadata csv in the same folder
        if (base/"forecast_temp_30d.pkl").exists():
            return base
        # nested layout (laptop): models/ subfolder
        if (base/"models"/"forecast_temp_30d.pkl").exists():
            return base/"models"
    return here

@st.cache_data
def find_meta_csv(base):
    # metadata city table — try flat names, then nested
    for cand in [base/"daily_forecast.csv.gz",
                 base/"daily_forecast_sample.csv.gz",
                 base.parent/"data"/"processed"/"daily_forecast.csv.gz",
                 Path(__file__).resolve().parent/"daily_forecast.csv.gz",
                 Path(__file__).resolve().parent/"daily_forecast_sample.csv.gz"]:
        if cand.exists():
            return cand
    return None

@st.cache_data
def load_meta(meta_path):
    df = pd.read_csv(meta_path, usecols=["city","latitude","longitude","coastal",
                     "climate_zone_id","continent_id","city_id"], low_memory=False)
    return df.drop_duplicates("city").set_index("city")

@st.cache_resource
def load_models(base):
    m={}
    for h in [30,60,90]:
        m[f"temp_{h}"]=joblib.load(base/f"forecast_temp_{h}d.pkl")
        for t in ["heatwave","rain","disaster"]:
            p=base/f"forecast_{t}_{h}d.pkl"
            if p.exists(): m[f"{t}_{h}"]=joblib.load(p)
    return m

def fetch_recent(lat, lon, retries=4):
    end=datetime.now().date(); start=end-timedelta(days=120)
    url=("https://archive-api.open-meteo.com/v1/archive"
         f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
         "&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,"
         "windspeed_10m_max,surface_pressure_mean&timezone=UTC")
    for a in range(retries):
        try:
            d=requests.get(url,timeout=90).json().get("daily",{})
            return pd.DataFrame({"date":pd.to_datetime(d.get("time",[])),
                "temp_mean":d.get("temperature_2m_mean",[]),
                "humidity":d.get("relative_humidity_2m_mean",[]),
                "precip_sum":d.get("precipitation_sum",[]),
                "wind_max":d.get("windspeed_10m_max",[]),
                "pressure":d.get("surface_pressure_mean",[])})
        except Exception:
            time.sleep(2*(a+1))
    return None

def build_features(df, meta):
    today=pd.Timestamp(datetime.now().date()); doy=today.dayofyear
    df=df.sort_values("date")
    f=lambda c,n: float(df[c].tail(n).mean()) if len(df) else np.nan
    fs=lambda c,n: float(df[c].tail(n).sum()) if len(df) else np.nan
    fm=lambda c,n: float(df[c].tail(n).max()) if len(df) else np.nan
    return pd.DataFrame([{
        "latitude":meta["latitude"],"longitude":meta["longitude"],"coastal":meta["coastal"],
        "climate_zone_id":meta["climate_zone_id"],"continent_id":meta["continent_id"],
        "city_id":meta["city_id"],
        "temp_mean_roll7":f("temp_mean",7),"temp_mean_roll30":f("temp_mean",30),"temp_mean_roll90":f("temp_mean",90),
        "humidity_roll7":f("humidity",7),"humidity_roll30":f("humidity",30),"humidity_roll90":f("humidity",90),
        "precip_sum_roll7":fs("precip_sum",7),"precip_sum_roll30":fs("precip_sum",30),"precip_sum_roll90":fs("precip_sum",90),
        "wind_max_roll7":fm("wind_max",7),"wind_max_roll30":fm("wind_max",30),"wind_max_roll90":fm("wind_max",90),
        "pressure_roll7":f("pressure",7),"pressure_roll30":f("pressure",30),"pressure_roll90":f("pressure",90),
        "doy_sin":np.sin(2*np.pi*doy/365.25),"doy_cos":np.cos(2*np.pi*doy/365.25),
        "month":today.month,"year":today.year,
    }])[FEATURES]

def predict_city(models, X):
    out={}
    for h in [30,60,90]:
        row={"temp":round(float(models[f"temp_{h}"].predict(X)[0]),1)}
        for t in ["heatwave","rain","disaster"]:
            mk=f"{t}_{h}"
            if mk in models:
                try: p=float(models[mk].predict_proba(X)[0,1])
                except: p=float(models[mk].predict(X)[0])
                row[t]=p
        out[h]=row
    return out

# ── App ────────────────────────────────────────────────────────
def main():
    base = find_base()
    meta_path = find_meta_csv(base)
    st.markdown('<div class="hdr">🌦️ Climate AI — Live Weather Forecast</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub">Forecasting 30 / 60 / 90 days ahead · 10 climate-diverse cities · '
                f'Generated {datetime.now().date()}</div>', unsafe_allow_html=True)

    if meta_path is None:
        st.error("City metadata file not found.")
        st.info(f"Looked next to the app in: `{base}`")
        st.info("Fix: upload `daily_forecast.csv.gz` (or `daily_forecast_sample.csv.gz`) "
                "to the same folder as app.py in your repo.")
        st.stop()
    try:
        meta=load_meta(meta_path); models=load_models(base)
    except Exception as e:
        st.error(f"Could not load models/metadata: {e}")
        st.info(f"Base folder: `{base}`\n\nMetadata: `{meta_path}`")
        st.info("Fix: make sure the 12 `forecast_*.pkl` files are uploaded next to app.py.")
        st.stop()

    with st.sidebar:
        st.markdown("### Select City")
        city=st.selectbox("City", CITIES, index=8)  # default Phoenix (most interesting)
        st.caption(f"Climate zone: **{ZONE[city]}**")
        st.markdown("---")
        st.caption("Temperature & heatwave = most reliable.\n\n"
                   "Rain 90-day = inherently hard (low confidence).\n\n"
                   "This is a live demo forecast; paper accuracy comes "
                   "from historical back-testing.")

    # fetch + predict for selected city
    with st.spinner(f"Fetching live weather & forecasting for {city}..."):
        m=meta.loc[city]
        recent=fetch_recent(m["latitude"], m["longitude"])
        if recent is None or recent.empty:
            st.error("Live weather fetch failed (network). Try again in a moment."); st.stop()
        X=build_features(recent, m)
        pred=predict_city(models, X)

    # ── headline cards ─────────────────────────────────────────
    exp(f"Below: the forecast for <b>{city}</b> at three lead times. "
        f"Temperature is the model's most reliable output. Alerts (heatwave / rain / "
        f"disaster) show the model's estimated probability.")
    cols=st.columns(3)
    for i,h in enumerate([30,60,90]):
        r=pred[h]; d=(datetime.now().date()+timedelta(days=h))
        with cols[i]:
            st.markdown(f'<div class="lbl">+{h} days · {d}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="big">{r["temp"]}°C</div>', unsafe_allow_html=True)
            hw=r.get("heatwave",0)
            if hw>=0.5: st.markdown(f'<span class="alert-hot">🔥 Heatwave likely ({hw:.0%})</span>', unsafe_allow_html=True)
            else:       st.markdown(f'<span class="alert-ok">No heatwave ({hw:.0%})</span>', unsafe_allow_html=True)
            rn=r.get("rain",0)
            st.write(f"🌧️ Heavy rain: **{rn:.0%}**" + ("  ⚠️" if rn>=0.5 else ""))
            ds=r.get("disaster",0)
            st.write(f"⚠️ Extreme event: **{ds:.0%}**" + ("  🚨" if ds>=0.5 else ""))

    st.markdown("---")

    # ── temperature trend chart (recent + forecast) ────────────
    st.markdown("#### Temperature: recent history → forecast")
    exp("The line shows the last 90 days of real temperature; the dots are the model's "
        "forecast at +30/+60/+90 days. This is how the near future is projected from recent trends.")
    hist=recent.sort_values("date").tail(90)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["temp_mean"], name="Recent (real)",
                             line=dict(color="#5b8fc9",width=2)))
    fdates=[datetime.now().date()+timedelta(days=h) for h in [30,60,90]]
    ftemps=[pred[h]["temp"] for h in [30,60,90]]
    fig.add_trace(go.Scatter(x=fdates, y=ftemps, name="Forecast", mode="markers+lines",
                             line=dict(color="#d98880",width=2,dash="dash"),
                             marker=dict(size=11,color="#d98880")))
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10,r=10,t=30,b=10),
                      yaxis_title="°C", legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # ── all-cities overview table ──────────────────────────────
    st.markdown("---")
    st.markdown("#### All 10 cities — 90-day outlook")
    exp("A quick comparison across every city at the 90-day horizon. Useful for seeing "
        "which climates the model flags as hot or high-risk.")
    if st.button("Generate all-cities overview (fetches live data for 10 cities)"):
        prog=st.progress(0.0); rows=[]
        for i,c in enumerate(CITIES):
            mm=meta.loc[c]; rec=fetch_recent(mm["latitude"],mm["longitude"])
            if rec is not None and not rec.empty:
                p=predict_city(models, build_features(rec,mm))[90]
                rows.append({"City":c,"Zone":ZONE[c],"Temp 90d (°C)":p["temp"],
                             "Heatwave":f'{p.get("heatwave",0):.0%}',
                             "Rain":f'{p.get("rain",0):.0%}',
                             "Disaster":f'{p.get("disaster",0):.0%}'})
            prog.progress((i+1)/len(CITIES)); time.sleep(0.5)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption("Climate AI · SMIU Karachi · LightGBM multi-horizon forecaster · "
               "Live data: Open-Meteo (ERA5). Demo forecast — not an official weather warning.")

if __name__=="__main__":
    main()
