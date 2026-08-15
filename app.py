#!/usr/bin/env python3
"""
Climate AI — Live Multi-Horizon Weather Forecast Dashboard
10 station-verified cities (from the 20-city published study)
4 forecast targets · 30/60/90-day horizons
Authors: Raiyan Sheikh, Syed Bilal — Supervisor: Syed Azeem Inam (SMIU Karachi)
"""
import streamlit as st
import pandas as pd, numpy as np, joblib, requests, time
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Climate AI — Live Forecast", page_icon="🌦️",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════
# THEME — colors chosen to work on BOTH dark and light mode
# ══════════════════════════════════════════════════════════════
CLR = {"temp":"#FF6B6B","heatwave":"#FFA94D","rain":"#4DABF7","disaster":"#9775FA",
       "grid":"rgba(128,128,128,.18)","txt":"rgba(128,128,128,.9)"}

st.markdown("""
<style>
@keyframes pulse { 0%{opacity:1} 50%{opacity:.55} 100%{opacity:1} }
.alert-live { animation: pulse 1.6s ease-in-out infinite; font-weight:700; }
.hdr { font-size:2rem; font-weight:800; text-align:center; padding:.2rem 0;
       background: linear-gradient(90deg,#FF6B6B,#FFA94D,#4DABF7,#9775FA);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub { text-align:center; opacity:.65; font-size:.9rem; margin-bottom:.8rem; }
.zone-chip { display:inline-block; padding:.15rem .6rem; border-radius:999px;
             border:1px solid rgba(128,128,128,.35); font-size:.78rem; opacity:.8; }
div[data-testid="stMetric"] { border:1px solid rgba(128,128,128,.25);
  border-radius:12px; padding:.6rem .8rem; }
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

# ERA5 reanalysis in the Open-Meteo archive lags real time by roughly 5 days.
# Requesting through "today" returns nulls at the tail, which silently corrupts
# the 7-day rolling features. We therefore end the window before the lag.
ERA5_LAG_DAYS = 6
HISTORY_DAYS  = 126   # 120 usable days after the lag is trimmed

# ── file discovery (flat repo OR laptop layout) ────────────────
@st.cache_data
def find_base():
    here = Path(__file__).resolve().parent
    for base in [here, Path.cwd(), Path.home()/"weather_v2"/"models"]:
        if (base/"forecast_temp_30d.pkl").exists(): return base
    return here

@st.cache_data
def find_meta(base):
    for c in [base/"daily_forecast.csv.gz", base/"daily_forecast_sample.csv.gz",
              base.parent/"data"/"processed"/"daily_forecast.csv.gz"]:
        if c.exists(): return c
    return None

@st.cache_data
def load_meta(p):
    df=pd.read_csv(p,usecols=["city","latitude","longitude","coastal",
        "climate_zone_id","continent_id","city_id"],low_memory=False)
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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent(lat, lon):
    """Fetch recent daily weather, ending before the ERA5 archive lag."""
    end   = datetime.now().date() - timedelta(days=ERA5_LAG_DAYS)
    start = end - timedelta(days=HISTORY_DAYS)
    url=("https://archive-api.open-meteo.com/v1/archive"
         f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
         "&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,"
         "windspeed_10m_max,surface_pressure_mean&timezone=UTC")
    for a in range(4):
        try:
            d=requests.get(url,timeout=90).json().get("daily",{})
            df=pd.DataFrame({"date":pd.to_datetime(d.get("time",[])),
                "temp_mean":d.get("temperature_2m_mean",[]),
                "humidity":d.get("relative_humidity_2m_mean",[]),
                "precip_sum":d.get("precipitation_sum",[]),
                "wind_max":d.get("windspeed_10m_max",[]),
                "pressure":d.get("surface_pressure_mean",[])})
            # Drop any unfilled tail so rolling windows are computed on real data
            df = df.dropna(subset=["temp_mean"]).reset_index(drop=True)
            return df if len(df) >= 30 else None
        except Exception: time.sleep(2*(a+1))
    return None

def build_X(df, m):
    # Forecast is ISSUED today, so seasonal encodings use today's date.
    # Only the weather history window is shifted back for the ERA5 lag.
    today=pd.Timestamp(datetime.now().date()); doy=today.dayofyear
    df=df.sort_values("date")
    f=lambda c,n: float(df[c].tail(n).mean()); fs=lambda c,n: float(df[c].tail(n).sum())
    fm=lambda c,n: float(df[c].tail(n).max())
    return pd.DataFrame([{ "latitude":m["latitude"],"longitude":m["longitude"],
        "coastal":m["coastal"],"climate_zone_id":m["climate_zone_id"],
        "continent_id":m["continent_id"],"city_id":m["city_id"],
        "temp_mean_roll7":f("temp_mean",7),"temp_mean_roll30":f("temp_mean",30),
        "temp_mean_roll90":f("temp_mean",90),
        "humidity_roll7":f("humidity",7),"humidity_roll30":f("humidity",30),
        "humidity_roll90":f("humidity",90),
        "precip_sum_roll7":fs("precip_sum",7),"precip_sum_roll30":fs("precip_sum",30),
        "precip_sum_roll90":fs("precip_sum",90),
        "wind_max_roll7":fm("wind_max",7),"wind_max_roll30":fm("wind_max",30),
        "wind_max_roll90":fm("wind_max",90),
        "pressure_roll7":f("pressure",7),"pressure_roll30":f("pressure",30),
        "pressure_roll90":f("pressure",90),
        "doy_sin":np.sin(2*np.pi*doy/365.25),"doy_cos":np.cos(2*np.pi*doy/365.25),
        "month":today.month,"year":today.year}])[FEATURES]

def predict(models, X):
    out={}
    for h in [30,60,90]:
        r={"temp":round(float(models[f"temp_{h}"].predict(X)[0]),1)}
        for t in ["heatwave","rain","disaster"]:
            k=f"{t}_{h}"
            if k in models:
                try: r[t]=float(models[k].predict_proba(X)[0,1])
                except: r[t]=float(models[k].predict(X)[0])
        out[h]=r
    return out

# ── chart builders (transparent bg = adapts to any theme) ──────
def base_layout(fig, h=330, title=None):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=h, margin=dict(l=10,r=10,t=70 if title else 40,b=10),
        title=dict(text=title or "", font=dict(size=14), x=0.01, xanchor="left",
                   y=0.97, yanchor="top"),
        font=dict(color=CLR["txt"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=CLR["grid"],zeroline=False),
        yaxis=dict(gridcolor=CLR["grid"],zeroline=False))
    return fig

def gauge(prob, color, label):
    fig=go.Figure(go.Indicator(mode="gauge+number", value=prob*100,
        number={"suffix":"%","font":{"size":34}},
        title={"text":label,"font":{"size":13}},
        gauge={"axis":{"range":[0,100],"tickcolor":CLR["txt"]},
               "bar":{"color":color,"thickness":.75},
               "bgcolor":"rgba(128,128,128,.08)",
               "threshold":{"line":{"color":color,"width":3},"value":50}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=210,
                      margin=dict(l=18,r=18,t=38,b=4),font=dict(color=CLR["txt"]))
    return fig

def horizon_bars(pred, key, color, ylab):
    hs=[30,60,90]; vals=[pred[h].get(key,0)*(1 if key=="temp" else 100) for h in hs]
    fig=go.Figure(go.Bar(x=[f"+{h} days" for h in hs], y=vals, marker_color=color,
        marker_line_width=0, text=[f"{v:.1f}" for v in vals], textposition="outside"))
    return base_layout(fig, 300, f"{ylab} across horizons")

def main():
    base=find_base(); meta_p=find_meta(base)
    st.markdown('<div class="hdr">Climate AI — Live Weather Forecast</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="sub">30 / 60 / 90-day forecasts · '
                f'10 station-verified cities (from the 20-city published study) · '
                f'Issued {datetime.now().strftime("%d %b %Y")}</div>',unsafe_allow_html=True)
    if meta_p is None:
        st.error("daily_forecast.csv.gz not found next to app.py — upload it to the repo."); st.stop()
    try:
        meta=load_meta(meta_p); models=load_models(base)
    except Exception as e:
        st.error(f"Model load failed: {e}"); st.stop()

    with st.sidebar:
        st.markdown("### 🌍 City")
        city=st.selectbox("Choose a city", CITIES, index=8, label_visibility="collapsed")
        st.markdown(f'<span class="zone-chip">Climate zone · {ZONE[city]}</span>',unsafe_allow_html=True)
        st.markdown("---")
        st.caption("**Reliability guide**\n\n"
                   "🌡️ Temperature — strongest signal\n\n"
                   "🔥 Heatwave — strong\n\n"
                   "⚠️ Disaster — moderate\n\n"
                   "🌧️ Rain — hardest at long range")
        st.markdown("---")
        st.caption("**Why 10 cities?**\n\n"
                   "The published study benchmarks 20 cities on ERA5 reanalysis. "
                   "This live demo covers the 10 that also passed station verification "
                   "(GHCN station within 50 km and at least 1,500 observation-days).")
        st.caption("Live data: Open-Meteo (ERA5) · Models: LightGBM · "
                   "Academic demo, not an official warning.")

    with st.spinner(f"Fetching live weather for {city} and forecasting…"):
        m=meta.loc[city]; rec=fetch_recent(float(m["latitude"]),float(m["longitude"]))
        if rec is None or rec.empty:
            st.error("Live fetch failed or returned too little data — try again shortly."); st.stop()
        pred=predict(models, build_X(rec,m))
        last_obs = rec["date"].max().date()

    # ── headline metrics: 4 parameter columns ──────────────────
    p30,p90=pred[30],pred[90]
    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.metric("🌡️ Temperature (+30d)", f'{p30["temp"]}°C',
                  delta=f'{p90["temp"]-p30["temp"]:+.1f}°C by +90d', delta_color="inverse")
    with c2:
        hw=p30.get("heatwave",0)
        st.metric("🔥 Heatwave risk (+30d)", f"{hw:.0%}",
                  delta="ALERT" if hw>=.5 else "low", delta_color="inverse" if hw>=.5 else "off")
    with c3:
        rn=p30.get("rain",0)
        st.metric("🌧️ Heavy-rain risk (+30d)", f"{rn:.0%}",
                  delta="likely" if rn>=.5 else "low", delta_color="inverse" if rn>=.5 else "off")
    with c4:
        ds=p30.get("disaster",0)
        st.metric("⚠️ Extreme-event risk (+30d)", f"{ds:.0%}",
                  delta="ALERT" if ds>=.5 else "low", delta_color="inverse" if ds>=.5 else "off")

    if max(p30.get("heatwave",0),p30.get("disaster",0))>=.5:
        st.markdown(f'<p class="alert-live" style="color:{CLR["heatwave"]};text-align:center">'
                    f'● LIVE ALERT — elevated extreme-weather risk for {city}</p>',unsafe_allow_html=True)

    st.caption(f"Latest available observation: {last_obs:%d %b %Y}. "
               f"ERA5 reanalysis publishes with a short lag, so the history window ends there; "
               f"forecast horizons are counted from today.")

    st.markdown("")

    # ── 4 tabs, 2 charts each ───────────────────────────────────
    t1,t2,t3,t4=st.tabs(["🌡️ Temperature","🔥 Heatwave","🌧️ Rain","⚠️ Disaster"])

    with t1:
        a,b=st.columns([3,2])
        with a:  # chart 1: history → forecast line
            hist=rec.sort_values("date").tail(90)
            fdates=[datetime.now().date()+timedelta(days=h) for h in [30,60,90]]
            ftemps=[pred[h]["temp"] for h in [30,60,90]]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=hist["date"],y=hist["temp_mean"],name="Recent observed",
                mode="lines",line=dict(color=CLR["rain"],width=2.4),
                fill="tozeroy",fillcolor="rgba(77,171,247,.08)"))
            fig.add_trace(go.Scatter(x=fdates,y=ftemps,name="Forecast",
                mode="lines+markers",line=dict(color=CLR["temp"],width=2.6,dash="dot"),
                marker=dict(size=12,symbol="diamond",line=dict(width=1,color="rgba(255,255,255,.6)"))))
            st.plotly_chart(base_layout(fig,360,"Recent temperature → 30/60/90-day forecast"),
                            use_container_width=True)
        with b:  # chart 2: horizon bars
            st.plotly_chart(horizon_bars(pred,"temp",CLR["temp"],"Forecast °C"),
                            use_container_width=True)

    with t2:
        a,b=st.columns([2,3])
        with a:  # chart 1: gauges
            st.plotly_chart(gauge(pred[30].get("heatwave",0),CLR["heatwave"],
                            "Heatwave probability (+30 days)"),use_container_width=True)
            st.plotly_chart(gauge(pred[90].get("heatwave",0),CLR["heatwave"],
                            "Heatwave probability (+90 days)"),use_container_width=True)
        with b:  # chart 2: probability across horizons + hot-day history
            hs=[30,60,90]; probs=[pred[h].get("heatwave",0)*100 for h in hs]
            fig=go.Figure(go.Scatter(x=[f"+{h}d" for h in hs],y=probs,mode="lines+markers",
                line=dict(color=CLR["heatwave"],width=3),
                marker=dict(size=14,line=dict(width=1,color="rgba(255,255,255,.6)")),
                fill="tozeroy",fillcolor="rgba(255,169,77,.12)"))
            fig.add_hline(y=50,line_dash="dash",line_color=CLR["temp"],
                          annotation_text="alert threshold")
            st.plotly_chart(base_layout(fig,300,"Heatwave probability by horizon (%)"),
                            use_container_width=True)
            hot=rec.copy(); hot["hot"]=(hot["temp_mean"]>hot["temp_mean"].quantile(.9)).astype(int)
            fig2=go.Figure(go.Bar(x=hot["date"],y=hot["hot"],marker_color=CLR["heatwave"],
                                  marker_line_width=0))
            fig2.update_yaxes(visible=False)
            st.plotly_chart(base_layout(fig2,180,"Recent unusually-hot days (top 10% of window)"),
                            use_container_width=True)

    with t3:
        a,b=st.columns([3,2])
        with a:  # chart 1: recent rainfall history
            fig=go.Figure(go.Bar(x=rec["date"],y=rec["precip_sum"],
                marker_color=CLR["rain"],marker_line_width=0))
            st.plotly_chart(base_layout(fig,340,"Daily rainfall — recent observed window (mm)"),
                            use_container_width=True)
        with b:  # chart 2: rain probability gauges
            st.plotly_chart(gauge(pred[30].get("rain",0),CLR["rain"],
                            "Heavy-rain probability (+30 days)"),use_container_width=True)
            st.plotly_chart(gauge(pred[90].get("rain",0),CLR["rain"],
                            "Heavy-rain probability (+90 days)"),use_container_width=True)
        st.caption("Long-range rainfall is the weakest target in this study "
                   "(F1 = 0.32 at 90 days). Treat these probabilities as exploratory, "
                   "not operational.")

    with t4:
        a,b=st.columns([2,3])
        with a:  # chart 1: disaster gauge
            st.plotly_chart(gauge(pred[30].get("disaster",0),CLR["disaster"],
                            "Extreme-event probability (+30 days)"),use_container_width=True)
            st.plotly_chart(gauge(pred[90].get("disaster",0),CLR["disaster"],
                            "Extreme-event probability (+90 days)"),use_container_width=True)
        with b:  # chart 2: all-risk comparison
            hs=[30,60,90]
            fig=go.Figure()
            for t,c in [("heatwave",CLR["heatwave"]),("rain",CLR["rain"]),("disaster",CLR["disaster"])]:
                fig.add_trace(go.Scatter(x=[f"+{h}d" for h in hs],
                    y=[pred[h].get(t,0)*100 for h in hs],name=t.capitalize(),
                    mode="lines+markers",line=dict(color=c,width=3),marker=dict(size=12)))
            fig.add_hline(y=50,line_dash="dash",line_color=CLR["txt"])
            st.plotly_chart(base_layout(fig,420,"All risk probabilities by horizon (%)"),
                            use_container_width=True)

    # ── all-cities comparison ───────────────────────────────────
    st.markdown("---")
    with st.expander("🌐 Compare all 10 cities (90-day outlook)"):
        st.caption("Fetches live data for every city. Run this once before a live "
                   "demo — results are cached for an hour.")
        if st.button("Run comparison — fetches live data for all cities (~30s)"):
            prog=st.progress(0.0,"Fetching…"); rows=[]
            for i,c in enumerate(CITIES):
                mm=meta.loc[c]; rc=fetch_recent(float(mm["latitude"]),float(mm["longitude"]))
                if rc is not None and not rc.empty:
                    p=predict(models,build_X(rc,mm))[90]
                    rows.append({"City":c,"Zone":ZONE[c],"Temp (°C)":p["temp"],
                        "Heatwave %":round(p.get("heatwave",0)*100),
                        "Rain %":round(p.get("rain",0)*100),
                        "Disaster %":round(p.get("disaster",0)*100)})
                prog.progress((i+1)/len(CITIES),f"{c} done")
            if not rows:
                st.warning("No cities returned data — check the connection and retry.")
            else:
                dfc=pd.DataFrame(rows)
                st.dataframe(
                    dfc, use_container_width=True, hide_index=True,
                    column_config={
                        "Temp (°C)": st.column_config.NumberColumn("Temp (°C)", format="%.1f°"),
                        "Heatwave %": st.column_config.ProgressColumn(
                            "Heatwave %", format="%d%%", min_value=0, max_value=100),
                        "Rain %": st.column_config.ProgressColumn(
                            "Rain %", format="%d%%", min_value=0, max_value=100),
                        "Disaster %": st.column_config.ProgressColumn(
                            "Disaster %", format="%d%%", min_value=0, max_value=100),
                    })
                fig=go.Figure(go.Bar(x=dfc["City"],y=dfc["Temp (°C)"],
                    marker=dict(color=dfc["Temp (°C)"],
                                colorscale=[[0,CLR["rain"]],[1,CLR["temp"]]]),
                    text=dfc["Temp (°C)"],textposition="outside"))
                st.plotly_chart(base_layout(fig,340,"90-day temperature forecast — all cities"),
                                use_container_width=True)

    st.caption("Climate AI · SMIU Karachi · LightGBM multi-horizon forecasting · "
               "Live data from Open-Meteo (ERA5) · Academic demo — not an official weather warning.")

if __name__=="__main__":
    main()
