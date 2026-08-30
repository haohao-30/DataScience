from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import (
    actual_vs_predicted_figure, forecast_figure, metric_by_horizon_figure, rmse_figure,
)
from src.constants import MODEL_NAMES, PREDICTORS, VISIBLE_FIELDS
from src.contracts import validate_contracts
from src.data_access import load_all_bundles, load_all_data
from src.feature_builder import (
    EXTERNAL_FIELDS, build_feature_row, latest_manual_defaults, validate_manual_inputs,
)
from src.prediction import historical_results, predict_manual, selected_models


st.set_page_config(page_title="Daily Gold Price Forecasting", page_icon="📈", layout="wide")

MODEL_COLORS = {"Ridge": "#2563EB", "KNN": "#F59E0B", "SVR": "#059669", "XGBoost": "#DC2626"}


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Validating saved coursework evidence…")
def cached_data() -> dict:
    return load_all_data()


@st.cache_resource(show_spinner="Loading four saved deployment models…")
def cached_bundles() -> dict:
    return load_all_bundles()


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def apply_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
        :root {
            color-scheme: light;
            --font-body:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            --font-display:'Sora','Inter',sans-serif;
            --accent:#1d4ed8;
            --accent-dark:#132f8f;
            --gold:#f2b93d;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1100px 480px at 92% -8%, rgba(29,78,216,.06), transparent 60%),
                radial-gradient(900px 420px at -6% 8%, rgba(242,185,61,.07), transparent 55%),
                #f4f7fb !important;
            color:#111827 !important;
            font-family:var(--font-body);
        }
        [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4,
        .page-title, .hero-title, .comparison-title, .section-heading-title, .kpi-value,
        .sb-name { font-family:var(--font-display) !important; }
        [data-testid="stHeader"] { background:rgba(244,247,251,.97) !important; }

        /* Custom scrollbar */
        ::-webkit-scrollbar { width:10px; height:10px; }
        ::-webkit-scrollbar-track { background:#eef1f6; }
        ::-webkit-scrollbar-thumb { background:#c3ccdb; border-radius:10px; border:2px solid #eef1f6; }
        ::-webkit-scrollbar-thumb:hover { background:#9fb0c9; }
        .block-container { max-width:1480px !important; padding-top:1.6rem !important; padding-bottom:3.6rem !important; }
        .block-container > div { animation:pageReveal .48s cubic-bezier(.22,1,.36,1) both; }
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] span { color:#111827 !important; opacity:1 !important; }
        hr { border-color:#e2e8f0 !important; margin:1.6rem 0 !important; }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background:
                radial-gradient(520px 260px at 100% 0%, rgba(242,185,61,.10), transparent 60%),
                radial-gradient(420px 320px at 0% 100%, rgba(37,99,235,.20), transparent 60%),
                linear-gradient(180deg,#0b1a33 0%,#0a1830 100%) !important;
            border-right:1px solid rgba(255,255,255,.06);
        }
        [data-testid="stSidebar"] > div:first-child { animation:sidebarReveal .48s cubic-bezier(.22,1,.36,1) both; }
        [data-testid="stSidebar"] * { color:#e2e8f0 !important; }
        [data-testid="stSidebar"] .sb-brand { display:flex; align-items:center; gap:.6rem; padding:.2rem 0 1rem 0; border-bottom:1px solid rgba(255,255,255,.12); margin-bottom:1.1rem; }
        [data-testid="stSidebar"] .sb-brand .sb-logo { font-size:1.6rem; }
        [data-testid="stSidebar"] .sb-brand .sb-name { font-weight:800; font-size:1.02rem; color:#ffffff !important; line-height:1.15; }
        [data-testid="stSidebar"] .sb-brand .sb-sub { font-size:.72rem; color:#93a4c3 !important; letter-spacing:.04em; text-transform:uppercase; }
        [data-testid="stSidebar"] .sb-section-title { font-size:.72rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:#7f93bb !important; margin:1.3rem 0 .5rem 0; }
        [data-testid="stSidebar"] .sb-fact { display:flex; justify-content:space-between; padding:.32rem 0; border-bottom:1px dashed rgba(255,255,255,.10); font-size:.86rem; }
        [data-testid="stSidebar"] .sb-fact span:first-child { color:#93a4c3 !important; }
        [data-testid="stSidebar"] .sb-fact span:last-child { font-weight:700; color:#f8fafc !important; }
        [data-testid="stSidebar"] .sb-legend-row { display:flex; align-items:center; gap:.5rem; padding:.24rem 0; font-size:.86rem; }
        [data-testid="stSidebar"] .sb-legend-row { transition:transform .18s ease, opacity .18s ease; }
        [data-testid="stSidebar"] .sb-legend-row:hover { transform:translateX(4px); }
        [data-testid="stSidebar"] .sb-dot { width:.6rem; height:.6rem; border-radius:50%; flex:0 0 auto; box-shadow:0 0 0 3px rgba(255,255,255,.06); }
        [data-testid="stSidebar"] .sb-badge { display:inline-block; padding:.28rem .6rem; border-radius:999px; background:rgba(37,99,235,.22); border:1px solid rgba(96,165,250,.35); color:#bfdbfe !important; font-size:.72rem; font-weight:750; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div { background:#152a4d !important; border-color:#2c4470 !important; color:#f8fafc !important; }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#cbd5e1 !important; font-weight:700 !important; }
        [data-testid="stSidebar"] [data-testid="stExpander"], [data-testid="stSidebar"] [data-testid="stExpander"] details { background:#10213e !important; border:1px solid rgba(255,255,255,.10) !important; }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary { background:transparent !important; }

        /* ---------- Inputs (main area) ---------- */
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stNumberInput"] input {
            background:#ffffff !important; color:#111827 !important; -webkit-text-fill-color:#111827 !important;
            border-color:#94a3b8 !important; opacity:1 !important;
        }
        [data-testid="stNumberInput"] input:disabled { background:#eef2f7 !important; color:#334155 !important; -webkit-text-fill-color:#334155 !important; }
        [data-testid="stNumberInput"] button { background:#eef2f7 !important; color:#111827 !important; }
        [data-testid="stNumberInput"] button svg { fill:#111827 !important; }
        [data-testid="stWidgetLabel"] p { color:#0f172a !important; font-weight:700 !important; }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#ffffff !important; color:#111827 !important; border:1px solid #94a3b8 !important; border-radius:9px !important; }
        [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
            border-color:#2563eb !important; box-shadow:0 0 0 3px rgba(37,99,235,.13) !important;
        }

        /* ---------- Tabs ---------- */
        [data-baseweb="tab-list"] { gap:.35rem; background:#e7edf6; padding:.3rem; border-radius:13px; width:fit-content; }
        button[data-baseweb="tab"] { border-radius:10px; padding:.5rem 1.05rem; transition:background .18s ease, transform .18s ease, box-shadow .18s ease; }
        button[data-baseweb="tab"]:hover { background:rgba(255,255,255,.68) !important; transform:translateY(-1px); }
        button[data-baseweb="tab"] p { color:#3a4a63 !important; font-weight:750 !important; }
        button[data-baseweb="tab"][aria-selected="true"] { background:#ffffff !important; box-shadow:0 3px 10px rgba(15,23,42,.10); }
        button[data-baseweb="tab"][aria-selected="true"] p { color:#1d4ed8 !important; }

        /* ---------- Hero ---------- */
        .hero { position:relative; overflow:hidden; padding:2.1rem 2.4rem; border:1px solid rgba(255,255,255,.14); border-radius:22px;
            background:linear-gradient(120deg,#071426 0%,#0e2a5e 52%,#1d4ed8 100%); background-size:140% 140%;
            box-shadow:0 16px 38px rgba(15,23,42,.20); color:#ffffff !important; margin-bottom:1.3rem;
            animation:heroReveal .65s cubic-bezier(.22,1,.36,1) both, heroGradient 14s ease-in-out infinite alternate; }
        .hero::before { content:""; position:absolute; width:300px; height:300px; right:-70px; top:-140px; border-radius:50%;
            background:radial-gradient(circle,rgba(251,191,36,.32),rgba(251,191,36,0) 68%); animation:glowFloat 8s ease-in-out infinite; }
        .hero::after { content:""; position:absolute; width:210px; height:210px; right:16%; bottom:-160px; border-radius:50%; background:rgba(96,165,250,.20); filter:blur(4px); animation:glowFloat 10s ease-in-out infinite reverse; }
        .hero-content { position:relative; z-index:2; }
        .hero-badge { display:inline-flex; align-items:center; padding:.34rem .72rem; border:1px solid rgba(255,255,255,.28); border-radius:999px;
            background:rgba(255,255,255,.10); color:#dbeafe !important; font-size:.76rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; }
        .hero-title { max-width:960px; margin-top:.9rem; color:#ffffff !important; font-size:clamp(2.1rem,3.9vw,3.55rem); font-weight:850; letter-spacing:-.04em; line-height:1.04; text-shadow:0 3px 16px rgba(0,0,0,.18); }
        .hero-title .gold {
            background:linear-gradient(100deg,#fde68a 0%,#fbbf24 32%,#f59e0b 55%,#fde68a 78%,#fbbf24 100%);
            background-size:220% auto;
            -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
            color:#fbbf24 !important; animation:goldShimmer 5s linear infinite;
        }
        .hero-subtitle { max-width:840px; margin-top:.85rem; color:#dbeafe !important; font-size:1.03rem; line-height:1.55; }
        .hero-tags { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.05rem; }
        .hero .hero-tag { padding:.38rem .7rem; border-radius:9px; background:rgba(15,23,42,.42); color:#ffffff !important; font-size:.8rem; font-weight:700; border:1px solid rgba(255,255,255,.16); }

        /* ---------- KPI strip ---------- */
        div[data-testid="stMetric"] { background:#ffffff !important; border:1px solid #dfe6ee; padding:.95rem 1rem; border-radius:14px;
            box-shadow:0 3px 10px rgba(15,23,42,.05); transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease;
            animation:cardReveal .52s cubic-bezier(.22,1,.36,1) both; }
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] { animation-delay:.07s; }
        div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] { animation-delay:.14s; }
        div[data-testid="stMetric"]:hover { transform:translateY(-3px); border-color:#bfdbfe; box-shadow:0 12px 26px rgba(30,64,175,.10); }
        [data-testid="stMetricLabel"] p { color:#5c6b82 !important; font-weight:750 !important; font-size:.8rem !important; letter-spacing:.02em; text-transform:uppercase; opacity:1 !important; }
        [data-testid="stMetricValue"] { color:#0f172a !important; opacity:1 !important; }
        [data-testid="stMetricValue"] > div { color:#0f172a !important; opacity:1 !important; }

        .overview-kpi { position:relative; overflow:hidden; min-height:142px; padding:1.05rem 1.15rem 1rem; background:#ffffff;
            border:1px solid #dfe6ee; border-radius:16px; box-shadow:0 6px 18px rgba(15,23,42,.055);
            transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease; animation:cardReveal .52s cubic-bezier(.22,1,.36,1) both; }
        .overview-kpi::before { content:""; position:absolute; inset:0 auto 0 0; width:4px; background:var(--kpi-accent,#1d4ed8); }
        .overview-kpi::after { content:""; position:absolute; width:120px; height:120px; right:-54px; top:-62px; border-radius:50%;
            background:color-mix(in srgb,var(--kpi-accent,#1d4ed8) 10%,transparent); }
        .overview-kpi:hover { transform:translateY(-3px); border-color:#bfdbfe; box-shadow:0 14px 30px rgba(30,64,175,.11); }
        .overview-kpi.kpi-delay-1 { animation-delay:.07s; }
        .overview-kpi.kpi-delay-2 { animation-delay:.14s; }
        .overview-kpi .kpi-top { position:relative; z-index:1; display:flex; justify-content:space-between; align-items:center; gap:.6rem; }
        .overview-kpi .kpi-label { color:#64748b !important; font-size:.73rem; font-weight:850; letter-spacing:.08em; text-transform:uppercase; }
        .overview-kpi .kpi-icon { display:flex; align-items:center; justify-content:center; width:2.15rem; height:2.15rem; border-radius:10px;
            background:color-mix(in srgb,var(--kpi-accent,#1d4ed8) 12%,#eff6ff); color:var(--kpi-accent,#1d4ed8) !important; font-size:1.05rem; }
        .overview-kpi .kpi-value { position:relative; z-index:1; margin-top:.35rem; color:#0f172a !important; font-size:clamp(1.75rem,2.7vw,2.45rem);
            font-weight:790; line-height:1.05; letter-spacing:-.035em; font-variant-numeric:tabular-nums; }
        .overview-kpi .kpi-note { position:relative; z-index:1; margin-top:.48rem; color:#64748b !important; font-size:.78rem; line-height:1.35; }
        .validation-ribbon { display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:.45rem .8rem; margin:.85rem 0 .25rem;
            padding:.62rem .9rem; border:1px solid #bbf7d0; border-radius:12px; background:linear-gradient(90deg,#ecfdf5,#f7fffb); color:#14532d !important;
            font-size:.78rem; font-weight:700; box-shadow:0 3px 10px rgba(5,150,105,.05); }
        .validation-ribbon * { color:#14532d !important; }
        .validation-ribbon .validation-dot { width:.52rem; height:.52rem; border-radius:50%; background:#10b981; box-shadow:0 0 0 4px rgba(16,185,129,.12); }
        .validation-ribbon .validation-separator { color:#86a995 !important; }

        /* ---------- Section headers ---------- */
        .section-heading { display:flex; align-items:flex-start; gap:.85rem; margin:1.7rem 0 .85rem 0; padding:.95rem 1.1rem; background:#ffffff;
            border:1px solid #dfe6ee; border-left:5px solid #1d4ed8; border-radius:13px; box-shadow:0 4px 14px rgba(15,23,42,.05);
            transition:transform .22s ease, box-shadow .22s ease; animation:sectionReveal .46s cubic-bezier(.22,1,.36,1) both; }
        .section-heading:hover { transform:translateY(-2px); box-shadow:0 10px 24px rgba(15,23,42,.08); }
        .section-heading .section-number { display:flex; align-items:center; justify-content:center; flex:0 0 2.3rem; height:2.3rem; border-radius:9px;
            background:#dbeafe; color:#1d4ed8 !important; font-size:.8rem; font-weight:850; letter-spacing:.02em; }
        .section-heading .section-heading-title { color:#0f172a !important; font-size:1.04rem; font-weight:850; line-height:1.25; margin:.02rem 0 .2rem 0; }
        .section-heading .section-heading-copy { color:#5c6b82 !important; font-size:.87rem; line-height:1.45; }

        .page-title-row { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:.6rem; margin:.1rem 0 1rem 0; }
        .page-title-row .page-title { font-size:1.28rem; font-weight:850; color:#0f172a !important; }
        .page-title-row .page-caption { font-size:.88rem; color:#5c6b82 !important; }

        .comparison-banner { margin:2.1rem 0 1.05rem 0; padding:1.3rem 1.45rem; border:1px solid #bfdbfe; border-radius:17px;
            background:linear-gradient(115deg,#eff6ff 0%,#ffffff 74%); box-shadow:0 7px 20px rgba(30,64,175,.07); }
        .comparison-banner .comparison-eyebrow { color:#1d4ed8 !important; font-size:.75rem; font-weight:850; letter-spacing:.1em; text-transform:uppercase; }
        .comparison-banner .comparison-title { color:#0f172a !important; font-size:1.55rem; font-weight:850; letter-spacing:-.02em; margin:.22rem 0 .28rem 0; }
        .comparison-banner .comparison-copy { color:#5c6b82 !important; font-size:.93rem; line-height:1.5; }

        /* ---------- Callout blocks ---------- */
        .winner { border:1px solid #f2b93d; background:linear-gradient(115deg,#fffbeb 0%,#fff7de 100%); border-radius:13px; padding:.75rem 1.05rem; color:#111827; box-shadow:0 4px 12px rgba(180,130,10,.08); }
        .winner, .winner div { color:#111827 !important; }
        .winner h3 { margin:0; color:#8a5a06 !important; font-size:1.0rem; }

        /* ---------- Featured forecast output ---------- */
        .forecast-result { position:relative; isolation:isolate; overflow:hidden; margin:.25rem 0 1.35rem; padding:1.2rem;
            border:1px solid rgba(37,99,235,.18); border-radius:20px;
            background:linear-gradient(145deg,#07172d 0%,#0d2d63 58%,#1748b3 100%);
            box-shadow:0 18px 42px rgba(15,36,82,.20); animation:resultPanelReveal .58s cubic-bezier(.22,1,.36,1) both; }
        .forecast-result::before { content:""; position:absolute; z-index:-1; width:270px; height:270px; right:-86px; top:-145px; border-radius:50%;
            background:radial-gradient(circle,rgba(251,191,36,.34),rgba(251,191,36,0) 70%); animation:resultGlow 5.5s ease-in-out infinite; }
        .forecast-result::after { content:""; position:absolute; z-index:-1; top:-70%; bottom:-70%; left:-36%; width:17%;
            background:linear-gradient(90deg,transparent,rgba(255,255,255,.10),transparent); transform:skewX(-19deg);
            animation:resultSweep 6.2s ease-in-out infinite; }
        .result-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; }
        .result-eyebrow { color:#93c5fd !important; font-size:.69rem; font-weight:850; letter-spacing:.11em; text-transform:uppercase; }
        .result-title { margin-top:.2rem; color:#ffffff !important; font-family:var(--font-display); font-size:1.35rem; font-weight:800; letter-spacing:-.025em; }
        .result-copy { margin-top:.25rem; color:#cbdcf7 !important; font-size:.78rem; line-height:1.4; }
        .result-model-badge { display:flex; align-items:center; gap:.65rem; padding:.58rem .75rem; border:1px solid rgba(255,220,102,.35);
            border-radius:12px; background:rgba(6,20,43,.42); box-shadow:inset 0 1px 0 rgba(255,255,255,.08); }
        .result-model-badge .badge-star { display:flex; align-items:center; justify-content:center; width:2rem; height:2rem; border-radius:9px;
            background:linear-gradient(145deg,#fff3a5,#f59e0b); color:#754400 !important; font-size:1rem; box-shadow:0 5px 12px rgba(245,158,11,.25); }
        .result-model-badge.selected { border-color:rgba(147,197,253,.36); }
        .result-model-badge.selected .badge-star { background:linear-gradient(145deg,#dbeafe,#60a5fa); color:#153e75 !important; box-shadow:0 5px 12px rgba(59,130,246,.24); }
        .result-model-badge .badge-label { color:#93a9cc !important; font-size:.58rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
        .result-model-badge .badge-value { margin-top:.08rem; color:#ffffff !important; font-family:var(--font-display); font-size:.95rem; font-weight:800; }
        .result-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.72rem; }
        .result-card { position:relative; overflow:hidden; min-height:104px; padding:.82rem .9rem; border:1px solid rgba(255,255,255,.12);
            border-radius:14px; background:rgba(255,255,255,.075); backdrop-filter:blur(8px); box-shadow:inset 0 1px 0 rgba(255,255,255,.07);
            transition:transform .2s ease,background .2s ease,border-color .2s ease; animation:resultCardReveal .48s cubic-bezier(.22,1,.36,1) both; }
        .result-card:nth-child(2) { animation-delay:.05s; } .result-card:nth-child(3) { animation-delay:.10s; }
        .result-card:nth-child(4) { animation-delay:.15s; } .result-card:nth-child(5) { animation-delay:.20s; }
        .result-card:nth-child(6) { animation-delay:.25s; }
        .result-card:hover { transform:translateY(-3px); background:rgba(255,255,255,.115); border-color:rgba(255,255,255,.24); }
        .result-card .result-label { color:#9fb5d7 !important; font-size:.64rem; font-weight:850; letter-spacing:.08em; text-transform:uppercase; }
        .result-card .result-value { margin-top:.48rem; color:#ffffff !important; font-family:var(--font-display); font-size:clamp(1.18rem,2vw,1.58rem);
            font-weight:760; letter-spacing:-.035em; line-height:1.08; font-variant-numeric:tabular-nums; }
        .result-card .result-note { margin-top:.28rem; color:#9fb5d7 !important; font-size:.64rem; }
        .result-card.primary { border-color:rgba(251,191,36,.46); background:linear-gradient(145deg,rgba(251,191,36,.20),rgba(245,158,11,.08)); }
        .result-card.primary::after { content:""; position:absolute; width:65%; height:180%; top:-42%; left:-82%;
            background:linear-gradient(90deg,transparent,rgba(255,250,213,.18),transparent); transform:skewX(-20deg); animation:priceCardShine 4.8s ease-in-out infinite; }
        .result-card.primary .result-label { color:#fde68a !important; }
        .result-card.primary .result-value { color:#fff4bd !important; text-shadow:0 0 18px rgba(251,191,36,.18); }
        .result-card.direction-up { background:linear-gradient(145deg,rgba(16,185,129,.20),rgba(5,150,105,.08)); border-color:rgba(52,211,153,.34); }
        .result-card.direction-down { background:linear-gradient(145deg,rgba(248,113,113,.20),rgba(220,38,38,.08)); border-color:rgba(248,113,113,.34); }
        .result-card.direction-neutral { background:linear-gradient(145deg,rgba(148,163,184,.18),rgba(100,116,139,.08)); }
        .direction-up .result-value { color:#6ee7b7 !important; } .direction-down .result-value { color:#fca5a5 !important; }
        .direction-neutral .result-value { color:#cbd5e1 !important; }
        .direction-signal { display:inline-flex; align-items:center; gap:.45rem; }
        .direction-signal .signal-dot { width:.52rem; height:.52rem; border-radius:50%; background:currentColor; box-shadow:0 0 0 0 currentColor; animation:signalPulse 2s ease-out infinite; }
        .status { border-left:6px solid #047857; background:#ecfdf5; color:#064e3b; padding:.78rem 1rem; border-radius:9px; }
        .status, .status * { color:#064e3b !important; font-weight:650; opacity:1 !important; }
        .note { background:#eef2ff; border-left:5px solid #4338ca; color:#1e1b4b; padding:.15rem 1rem; border-radius:9px; }
        .note, .note * { color:#1e1b4b !important; opacity:1 !important; }
        .note-title { font-weight:800; padding-top:.75rem; font-size:.86rem; }
        .note-grid { display:flex; flex-wrap:wrap; gap:1.6rem; padding:.35rem 0 .8rem 0; }
        .note-item .note-label { font-size:.78rem; color:#4338ca !important; opacity:.85 !important; }
        .note-item .note-value { font-size:.98rem; font-weight:750; font-variant-numeric:tabular-nums; }
        .note-foot { font-size:.78rem; opacity:.75 !important; padding-bottom:.7rem; }

        button[kind="primary"], [data-testid="stFormSubmitButton"] button { background:#1d4ed8 !important; border-color:#1d4ed8 !important; border-radius:10px !important;
            box-shadow:0 5px 14px rgba(29,78,216,.18); transition:transform .18s ease, box-shadow .18s ease, background .18s ease; }
        button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-2px); background:#1e40af !important; box-shadow:0 10px 22px rgba(29,78,216,.24); }
        button[kind="primary"]:active, [data-testid="stFormSubmitButton"] button:active { transform:translateY(0); }
        button[kind="primary"] p, [data-testid="stFormSubmitButton"] button p,
        button[kind="primary"] span, [data-testid="stFormSubmitButton"] button span { color:#ffffff !important; font-weight:750 !important; }

        [data-testid="stExpander"], [data-testid="stExpander"] details { background:#ffffff !important; border:1px solid #dfe6ee !important; border-radius:13px !important; overflow:hidden; }
        [data-testid="stExpander"] summary { background:#eef2f7 !important; color:#0f172a !important; padding:.72rem 1rem !important; }
        [data-testid="stExpander"] summary:hover { background:#e2e9f4 !important; }
        [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color:#0f172a !important; font-weight:750 !important; opacity:1 !important; }
        [data-testid="stExpander"] summary svg { fill:#1d4ed8 !important; color:#1d4ed8 !important; }
        [data-testid="stExpanderDetails"] { background:#ffffff !important; color:#111827 !important; padding:.7rem 1rem 1rem !important; }

        .table-wrap { overflow-x:auto; margin:.4rem 0 1.2rem 0; border:1px solid #e5e9f0; border-radius:12px; }
        .data-table { border-collapse:collapse; width:100%; background:white; color:#111827; font-size:.9rem; }
        .data-table th { position:sticky; top:0; z-index:1; background:#0f172a; color:white; padding:.62rem .68rem; text-align:left; white-space:nowrap; }
        .data-table td { border-bottom:1px solid #e5e9f0; padding:.55rem .68rem; white-space:nowrap; }
        .data-table tr:nth-child(even) { background:#f6f8fb; }
        .data-table tr:hover { background:#eef4ff; }

        .app-footer { margin-top:2.6rem; padding-top:1.2rem; border-top:1px solid #e2e8f0; display:flex; justify-content:space-between;
            flex-wrap:wrap; gap:.4rem; color:#7c8aa0 !important; font-size:.82rem; }
        .app-footer * { color:#7c8aa0 !important; }
        .app-footer .footer-dot { color:#c7cedb !important; padding:0 .5rem; }

        /* ---------- Plotly chart cards ---------- */
        [data-testid="stPlotlyChart"] {
            background:#ffffff; border:1px solid #e3e8f0; border-radius:16px; padding:.7rem .6rem .2rem;
            box-shadow:0 6px 18px rgba(15,23,42,.05); margin-bottom:.4rem;
            transition:box-shadow .22s ease, border-color .22s ease;
        }
        [data-testid="stPlotlyChart"]:hover { box-shadow:0 14px 30px rgba(15,23,42,.09); border-color:#cfdaea; }

        /* ---------- Alerts ---------- */
        [data-testid="stAlertContentInfo"], [data-testid="stAlertContentError"], [data-testid="stAlertContentSuccess"] { font-weight:600 !important; }
        div[data-testid="stAlert"] { border-radius:12px !important; border:1px solid transparent; box-shadow:0 4px 12px rgba(15,23,42,.05); }

        /* ---------- Pill radio (results view toggle) ---------- */
        [data-testid="stRadio"] > div { gap:.4rem; background:#eef2f7; padding:.28rem; border-radius:999px; width:fit-content; }
        [data-testid="stRadio"] label { background:transparent; border-radius:999px; padding:.32rem .85rem !important; transition:background .18s ease, box-shadow .18s ease; }
        [data-testid="stRadio"] label:has(input:checked) { background:#ffffff; box-shadow:0 2px 8px rgba(15,23,42,.10); }
        [data-testid="stRadio"] label p { font-weight:700 !important; font-size:.86rem !important; }

        /* ---------- Spinner ---------- */
        [data-testid="stSpinner"] > div { border-top-color:var(--accent) !important; }

        /* ---------- Tab panel entrance ---------- */
        [data-testid="stTabsPanel"] { animation:panelFade .38s ease both; }

        /* ---------- Subtle falling 3D gold bars ---------- */
        .gold-rain {
            position:fixed; inset:0; overflow:hidden; pointer-events:none; z-index:999;
            user-select:none; contain:strict;
        }
        .gold-rain .gold-bar {
            position:absolute; top:-11vh; left:var(--left); width:62px; height:28px;
            opacity:0; will-change:transform,opacity;
            filter:drop-shadow(0 8px 8px rgba(73,46,0,.28)) drop-shadow(0 0 10px rgba(255,199,38,.14));
            animation:goldBarFall var(--duration) linear infinite; animation-delay:var(--delay);
        }
        .gold-rain .gold-bar:nth-child(1) { --duration:10.8s !important; }
        .gold-rain .gold-bar:nth-child(2) { --duration:13.4s !important; }
        .gold-rain .gold-bar:nth-child(3) { --duration:11.6s !important; }
        .gold-rain .gold-bar:nth-child(4) { --duration:14.8s !important; }
        .gold-rain .gold-bar:nth-child(5) { --duration:12.2s !important; }
        .gold-rain .gold-bar:nth-child(6) { --duration:13.9s !important; }
        .gold-rain .gold-bar:nth-child(7) { --duration:12.7s !important; }
        .gold-rain .gold-bar:nth-child(8) { --duration:11.2s !important; }
        .gold-rain .gold-bar::after {
            content:""; position:absolute; z-index:-1; left:50%; top:-43px; width:3px; height:48px; border-radius:99px;
            background:linear-gradient(180deg,transparent 0%,rgba(251,191,36,.03) 22%,rgba(245,158,11,.35) 100%);
            transform:translateX(-50%); filter:blur(1.2px); opacity:0;
            animation:goldSpeedTrail var(--duration) linear infinite; animation-delay:var(--delay);
        }
        .gold-rain .gold-bar .gold-solid {
            position:absolute; inset:0; display:block; transform-style:preserve-3d;
            animation:goldBarSpin var(--duration) linear infinite; animation-delay:var(--delay);
        }
        .gold-rain .gold-bar .bar-face {
            position:absolute; display:block; box-sizing:border-box; overflow:hidden;
            transform-style:preserve-3d; backface-visibility:visible;
            border:1px solid rgba(118,71,0,.38);
        }
        .gold-rain .gold-bar .bar-front,
        .gold-rain .gold-bar .bar-back {
            left:0; top:0; width:62px; height:28px; border-radius:3px;
        }
        .gold-rain .gold-bar .bar-front {
            transform:translateZ(9px);
            background:linear-gradient(145deg,#fff3a8 0%,#ffd44f 15%,#e7a816 40%,#bd7903 68%,#7c4700 100%);
            box-shadow:inset 0 3px 4px rgba(255,255,221,.55),inset 0 -5px 6px rgba(76,41,0,.35);
        }
        .gold-rain .gold-bar .bar-back {
            transform:rotateY(180deg) translateZ(9px);
            background:linear-gradient(215deg,#f8c83e,#b46e00 48%,#6f3d00 100%);
            box-shadow:inset 0 3px 5px rgba(255,225,100,.35),inset 0 -4px 6px rgba(68,36,0,.38);
        }
        .gold-rain .gold-bar .bar-top,
        .gold-rain .gold-bar .bar-bottom {
            left:0; top:5px; width:62px; height:18px; border-radius:3px;
        }
        .gold-rain .gold-bar .bar-top {
            transform:rotateX(90deg) translateZ(14px);
            background:linear-gradient(110deg,#fffbd6 0%,#ffe778 18%,#ffc72e 43%,#efad0c 64%,#ffdc5c 84%,#a96800 100%);
            box-shadow:inset 0 2px 4px rgba(255,255,237,.82),inset 0 -3px 4px rgba(124,72,0,.23);
        }
        .gold-rain .gold-bar .bar-bottom {
            transform:rotateX(-90deg) translateZ(14px);
            background:linear-gradient(90deg,#6d3b00,#a96500 55%,#784200);
        }
        .gold-rain .gold-bar .bar-left,
        .gold-rain .gold-bar .bar-right {
            left:22px; top:0; width:18px; height:28px; border-radius:2px;
        }
        .gold-rain .gold-bar .bar-left {
            transform:rotateY(-90deg) translateZ(31px);
            background:linear-gradient(160deg,#e5a511,#965800 58%,#633600 100%);
        }
        .gold-rain .gold-bar .bar-right {
            transform:rotateY(90deg) translateZ(31px);
            background:linear-gradient(200deg,#ffe16a,#d38e05 45%,#7a4500 100%);
        }
        .gold-rain .gold-bar .bar-top::after {
            content:""; position:absolute; top:-45%; bottom:-45%; width:42%; left:-65%;
            background:linear-gradient(90deg,transparent,rgba(255,255,246,.82),transparent);
            transform:skewX(-22deg); animation:barGleam 4.2s ease-in-out infinite; animation-delay:var(--gleam);
        }
        .gold-rain .gold-bar .bar-stamp {
            position:absolute; left:8px; right:8px; top:5px; text-align:center;
            color:#754400 !important; font-family:Georgia,serif; font-size:8px; font-weight:900;
            letter-spacing:.12em; line-height:1; text-shadow:0 1px 0 rgba(255,247,177,.92); opacity:.88;
        }
        .gold-rain .gold-bar .bar-purity {
            position:absolute; left:8px; right:8px; top:15px; text-align:center;
            color:#8b5400 !important; font-family:Georgia,serif; font-size:5px; font-style:normal; font-weight:700;
            letter-spacing:.08em; line-height:1; opacity:.72;
        }

        /* ---------- Refined motion ---------- */
        @keyframes goldShimmer {
            to { background-position:-220% center; }
        }
        @keyframes panelFade {
            from { opacity:0; transform:translateY(6px); }
            to { opacity:1; transform:translateY(0); }
        }
        @keyframes pageReveal {
            from { opacity:0; transform:translateY(8px); }
            to { opacity:1; transform:translateY(0); }
        }
        @keyframes sidebarReveal {
            from { opacity:0; transform:translateX(-12px); }
            to { opacity:1; transform:translateX(0); }
        }
        @keyframes heroReveal {
            from { opacity:0; transform:translateY(14px) scale(.995); }
            to { opacity:1; transform:translateY(0) scale(1); }
        }
        @keyframes heroGradient {
            from { background-position:0% 50%; }
            to { background-position:100% 50%; }
        }
        @keyframes glowFloat {
            0%,100% { transform:translate3d(0,0,0) scale(1); opacity:.82; }
            50% { transform:translate3d(-18px,14px,0) scale(1.08); opacity:1; }
        }
        @keyframes cardReveal {
            from { opacity:0; transform:translateY(12px); }
            to { opacity:1; transform:translateY(0); }
        }
        @keyframes sectionReveal {
            from { opacity:0; transform:translateY(10px); }
            to { opacity:1; transform:translateY(0); }
        }
        @keyframes resultPanelReveal {
            from { opacity:0; transform:translateY(16px) scale(.994); }
            to { opacity:1; transform:translateY(0) scale(1); }
        }
        @keyframes resultCardReveal {
            from { opacity:0; transform:translateY(12px) scale(.98); }
            to { opacity:1; transform:translateY(0) scale(1); }
        }
        @keyframes resultGlow {
            0%,100% { transform:translate3d(0,0,0) scale(1); opacity:.78; }
            50% { transform:translate3d(-16px,14px,0) scale(1.09); opacity:1; }
        }
        @keyframes resultSweep {
            0%,56% { left:-36%; opacity:0; }
            66% { opacity:.75; }
            82%,100% { left:118%; opacity:0; }
        }
        @keyframes priceCardShine {
            0%,58% { left:-82%; opacity:0; }
            68% { opacity:.9; }
            84%,100% { left:135%; opacity:0; }
        }
        @keyframes signalPulse {
            0% { box-shadow:0 0 0 0 currentColor; opacity:1; }
            75%,100% { box-shadow:0 0 0 8px transparent; opacity:.9; }
        }
        @keyframes goldBarFall {
            0% { opacity:0; transform:translate3d(0,-14vh,0) scale3d(var(--scale),var(--scale),var(--scale)); }
            8% { opacity:.76; }
            20% { transform:translate3d(var(--drift),-6vh,0) scale3d(var(--scale),var(--scale),var(--scale)); }
            40% { transform:translate3d(0,10vh,0) scale3d(var(--scale),var(--scale),var(--scale)); }
            60% { transform:translate3d(var(--drift),36vh,0) scale3d(var(--scale),var(--scale),var(--scale)); }
            80% { transform:translate3d(0,74vh,0) scale3d(var(--scale),var(--scale),var(--scale)); }
            92% { opacity:.70; }
            100% { opacity:0; transform:translate3d(var(--drift),118vh,0) scale3d(var(--scale),var(--scale),var(--scale)); }
        }
        @keyframes goldBarSpin {
            0% { transform:perspective(900px) rotateX(28deg) rotateY(-22deg) rotateZ(-10deg); }
            20% { transform:perspective(900px) rotateX(61deg) rotateY(18deg) rotateZ(17deg); }
            40% { transform:perspective(900px) rotateX(132deg) rotateY(91deg) rotateZ(61deg); }
            60% { transform:perspective(900px) rotateX(244deg) rotateY(202deg) rotateZ(127deg); }
            80% { transform:perspective(900px) rotateX(391deg) rotateY(352deg) rotateZ(221deg); }
            100% { transform:perspective(900px) rotateX(574deg) rotateY(548deg) rotateZ(342deg); }
        }
        @keyframes goldSpeedTrail {
            0%,42% { opacity:0; height:18px; }
            62% { opacity:.12; height:30px; }
            82% { opacity:.34; height:48px; }
            94% { opacity:.46; height:62px; }
            100% { opacity:0; height:70px; }
        }
        @keyframes barGleam {
            0%,58% { left:-65%; opacity:0; }
            67% { opacity:.92; }
            84%,100% { left:125%; opacity:0; }
        }

        @media (prefers-reduced-motion: reduce) {
            .gold-rain { display:none !important; }
            *, *::before, *::after {
                animation-duration:.01ms !important;
                animation-iteration-count:1 !important;
                scroll-behavior:auto !important;
                transition-duration:.01ms !important;
            }
        }

        @media (max-width: 760px) {
            .block-container { padding-top:.8rem !important; padding-left:1rem !important; padding-right:1rem !important; }
            .hero { padding:1.45rem 1.25rem; border-radius:17px; }
            .hero-title { font-size:2.1rem; }
            .hero-subtitle { font-size:.92rem; }
            .overview-kpi { min-height:126px; }
            .section-heading { padding:.8rem .85rem; }
            .validation-ribbon { justify-content:flex-start; }
            .forecast-result { padding:.9rem; border-radius:16px; }
            .result-grid { grid-template-columns:1fr; }
            .result-card { min-height:88px; }
            .gold-rain .gold-bar:nth-child(n+6) { display:none; }
        }

        /* ---------- Final sidebar contrast safeguards ---------- */
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] p,
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] span,
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] label,
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] li,
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] strong {
            color:#e2e8f0 !important; -webkit-text-fill-color:#e2e8f0 !important; opacity:1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] .sb-legend-row span:last-child {
            color:#f8fafc !important; -webkit-text-fill-color:#f8fafc !important; font-weight:700 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] .sb-section-title {
            color:#9fb3d9 !important; -webkit-text-fill-color:#9fb3d9 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background:#10213e !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] [data-testid="stExpanderDetails"] p,
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] [data-testid="stExpanderDetails"] li,
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] [data-testid="stExpanderDetails"] strong {
            color:#e2e8f0 !important; -webkit-text-fill-color:#e2e8f0 !important; opacity:1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
            fill:#bfdbfe !important; color:#bfdbfe !important;
        }
        </style>
        <div class="gold-rain" aria-hidden="true">
          <span class="gold-bar" style="--left:20%;--delay:-3s;--duration:21s;--scale:.82;--drift:34px;--gleam:-1s"><span class="gold-solid"><i class="bar-face bar-front"><b class="bar-stamp">GOLD</b><em class="bar-purity">999.9</em></i><i class="bar-face bar-back"></i><i class="bar-face bar-top"></i><i class="bar-face bar-bottom"></i><i class="bar-face bar-left"></i><i class="bar-face bar-right"></i></span></span>
          <span class="gold-bar" style="--left:31%;--delay:-15s;--duration:26s;--scale:.64;--drift:-42px;--gleam:-3s"><span class="gold-solid"><i class="bar-face bar-front"><b class="bar-stamp">GOLD</b><em class="bar-purity">999.9</em></i><i class="bar-face bar-back"></i><i class="bar-face bar-top"></i><i class="bar-face bar-bottom"></i><i class="bar-face bar-left"></i><i class="bar-face bar-right"></i></span></span>
          <span class="gold-bar" style="--left:42%;--delay:-8s;--duration:23s;--scale:1.00;--drift:48px;--gleam:-2s"><span class="gold-solid"><i class="bar-face bar-front"><b class="bar-stamp">GOLD</b><em class="bar-purity">999.9</em></i><i class="bar-face bar-back"></i><i class="bar-face bar-top"></i><i class="bar-face bar-bottom"></i><i class="bar-face bar-left"></i><i class="bar-face bar-right"></i></span></span>
          <span class="gold-bar" style="--left:54%;--delay:-21s;--duration:29s;--scale:.68;--drift:-28px;--gleam:-4s"><span class="gold-solid"><i class="bar-face bar-front"><b class="bar-stamp">GOLD</b><em class="bar-purity">999.9</em></i><i class="bar-face bar-back"></i><i class="bar-face bar-top"></i><i class="bar-face bar-bottom"></i><i class="bar-face bar-left"></i><i class="bar-face bar-right"></i></span></span>
          <span class="gold-bar" style="--left:66%;--delay:-11s;--duration:24s;--scale:.78;--drift:40px;--gleam:-.5s"><span class="gold-solid"><i class="bar-face bar-front"><b class="bar-stamp">GOLD</b><em class="bar-purity">999.9</em></i><i class="bar-face bar-back"></i><i class="bar-face bar-top"></i><i class="bar-face bar-bottom"></i><i class="bar-face bar-left"></i><i class="bar-face bar-right"></i></span></span>
          <span class="gold-bar" style="--left:77%;--delay:-1s;--duration:27s;--scale:.60;--drift:-36px;--gleam:-2.5s"><span class="gold-solid"><i class="bar-face bar-front"><b class="bar-stamp">GOLD</b><em class="bar-purity">999.9</em></i><i class="bar-face bar-back"></i><i class="bar-face bar-top"></i><i class="bar-face bar-bottom"></i><i class="bar-face bar-left"></i><i class="bar-face bar-right"></i></span></span>
          <span class="gold-bar" style="--left:87%;--delay:-18s;--duration:28s;--scale:.88;--drift:44px;--gleam:-3.5s"><span class="gold-solid"><i class="bar-face bar-front"><b class="bar-stamp">GOLD</b><em class="bar-purity">999.9</em></i><i class="bar-face bar-back"></i><i class="bar-face bar-top"></i><i class="bar-face bar-bottom"></i><i class="bar-face bar-left"></i><i class="bar-face bar-right"></i></span></span>
          <span class="gold-bar" style="--left:96%;--delay:-6s;--duration:24s;--scale:.66;--drift:-24px;--gleam:-1.5s"><span class="gold-solid"><i class="bar-face bar-front"><b class="bar-stamp">GOLD</b><em class="bar-purity">999.9</em></i><i class="bar-face bar-back"></i><i class="bar-face bar-top"></i><i class="bar-face bar-bottom"></i><i class="bar-face bar-left"></i><i class="bar-face bar-right"></i></span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Small render helpers
# --------------------------------------------------------------------------- #
def section_header(number: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="section-number">{number}</div>
          <div>
            <div class="section-heading-title">{title}</div>
            <div class="section-heading-copy">{description}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="page-title-row">
          <div class="page-title">{title}</div>
        </div>
        <div class="page-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def format_results(frame: pd.DataFrame, historical: bool = False) -> pd.DataFrame:
    columns = [
        "Model", "Horizon", "Current Price", "Predicted Return",
        "Predicted Return Percentage", "Predicted Price Change", "Predicted Price", "Direction",
    ]
    if historical:
        columns += [
            "Target Date", "Actual Return (revealed after prediction)",
            "Actual Price (revealed after prediction)", "Persistence Price",
        ]
    return frame[columns].copy()


def show_result_table(frame: pd.DataFrame, historical: bool = False) -> None:
    display = format_results(frame, historical)
    formats = {
        "Current Price": "{:,.2f}",
        "Predicted Return": "{:.6f}",
        "Predicted Return Percentage": "{:.3f}%",
        "Predicted Price Change": "{:+,.2f}",
        "Predicted Price": "{:,.2f}",
    }
    if historical:
        formats.update({
            "Target Date": lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
            "Actual Return (revealed after prediction)": "{:.6f}",
            "Actual Price (revealed after prediction)": "{:,.2f}",
            "Persistence Price": "{:,.2f}",
        })
    styled = display.style.format(formats).map(
        lambda value: "color:#047857;font-weight:700" if value == "Up" else (
            "color:#b91c1c;font-weight:700" if value == "Down" else "color:#4b5563;font-weight:700"
        ), subset=["Direction"]
    ).hide(axis="index").set_table_attributes('class="data-table"')
    st.markdown(f'<div class="table-wrap">{styled.to_html()}</div>', unsafe_allow_html=True)


def show_result_browser(frame: pd.DataFrame, historical: bool, key_prefix: str) -> None:
    view = st.radio(
        "Results view",
        ["Show one horizon", "Show all horizons"],
        horizontal=True,
        key=f"{key_prefix}_results_view",
    )

    if view == "Show all horizons":
        show_result_table(frame, historical=historical)
        return

    horizon_options = frame["Horizon"].drop_duplicates().tolist()
    chosen_horizon = st.selectbox(
        "Horizon",
        horizon_options,
        key=f"{key_prefix}_result_horizon",
    )
    selected = frame.loc[frame["Horizon"].eq(chosen_horizon)]
    show_result_table(selected, historical=historical)


def show_plain_table(frame: pd.DataFrame, formats: dict | None = None) -> None:
    styled = frame.style.hide(axis="index").set_table_attributes('class="data-table"')
    if formats:
        styled = styled.format(formats)
    st.markdown(f'<div class="table-wrap">{styled.to_html()}</div>', unsafe_allow_html=True)


def show_featured(featured: pd.DataFrame, display_model: str, official_best_model: str) -> None:
    h1 = featured.loc[(featured["Model"].eq(display_model)) & (featured["Horizon"].eq("H1"))].iloc[0]
    is_official_winner = display_model == official_best_model
    result_eyebrow = (
        "Forecast generated · Official winner · Direct H1 prediction"
        if is_official_winner else "Forecast generated · Selected model · Direct H1 prediction"
    )
    result_copy = (
        "Official Evaluation winner · Each horizon is predicted directly, not recursively."
        if is_official_winner else
        f"Showing the model selected in the sidebar · Official overall Evaluation winner: {official_best_model}."
    )
    badge_class = "" if is_official_winner else " selected"
    badge_icon = "★" if is_official_winner else "◆"
    badge_label = "Best overall model" if is_official_winner else "Selected model"
    direction = str(h1["Direction"])
    direction_key = direction.strip().lower().replace(" ", "-")
    direction_class = {
        "up": "direction-up", "down": "direction-down", "no-change": "direction-neutral",
    }.get(direction_key, "direction-neutral")
    direction_icon = {"up": "↗", "down": "↘", "no-change": "→"}.get(direction_key, "→")
    price_change = float(h1["Predicted Price Change"])
    price_change_text = f'{"+" if price_change >= 0 else "−"}₹{abs(price_change):,.2f}'

    st.markdown(
        f"""
        <div class="forecast-result">
          <div class="result-header">
            <div>
              <div class="result-eyebrow">{result_eyebrow}</div>
              <div class="result-title">{display_model} Forecast Snapshot</div>
              <div class="result-copy">{result_copy}</div>
            </div>
            <div class="result-model-badge{badge_class}">
              <div class="badge-star">{badge_icon}</div>
              <div><div class="badge-label">{badge_label}</div><div class="badge-value">{display_model}</div></div>
            </div>
          </div>
          <div class="result-grid">
            <div class="result-card">
              <div class="result-label">Current Price</div>
              <div class="result-value">₹{h1["Current Price"]:,.2f}</div>
              <div class="result-note">INR per 10g · forecast origin</div>
            </div>
            <div class="result-card">
              <div class="result-label">H1 Predicted Return</div>
              <div class="result-value">{h1["Predicted Return"]:+.6f}</div>
              <div class="result-note">Direct cumulative return</div>
            </div>
            <div class="result-card">
              <div class="result-label">H1 Return Percentage</div>
              <div class="result-value">{h1["Predicted Return Percentage"]:+.3f}%</div>
              <div class="result-note">Relative to current price</div>
            </div>
            <div class="result-card">
              <div class="result-label">Predicted Price Change</div>
              <div class="result-value">{price_change_text}</div>
              <div class="result-note">Predicted price minus current price</div>
            </div>
            <div class="result-card primary">
              <div class="result-label">H1 Predicted Next Price</div>
              <div class="result-value">₹{h1["Predicted Price"]:,.2f}</div>
              <div class="result-note">Reconstructed in original price units</div>
            </div>
            <div class="result-card {direction_class}">
              <div class="result-label">Predicted Direction</div>
              <div class="result-value direction-signal"><span class="signal-dot"></span>{direction_icon} {direction}</div>
              <div class="result-note">Based on the predicted return sign</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(data: dict, contracts: dict) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sb-brand">
              <div class="sb-logo">📈</div>
              <div>
                <div class="sb-name">Gold Forecast Studio</div>
                <div class="sb-sub">BMDS2003 · Data Science</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section-title">Forecast display</div>', unsafe_allow_html=True)
        selection = st.selectbox(
            "Model shown in the detailed tables and charts",
            ["All Models", *MODEL_NAMES],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-section-title">Model legend</div>', unsafe_allow_html=True)
        for name in MODEL_NAMES:
            crown = " 👑" if name == contracts["best_model"] else ""
            st.markdown(
                f'<div class="sb-legend-row"><span class="sb-dot" style="background:{MODEL_COLORS[name]}"></span>'
                f'<span>{name}{crown}</span></div>',
                unsafe_allow_html=True,
            )

        canonical = data["canonical"]
        st.markdown('<div class="sb-section-title">Dataset snapshot</div>', unsafe_allow_html=True)
        facts = [
            ("Total rows", f'{len(canonical):,}'),
            ("Evaluation dates", f'{len(contracts["common_dates"]):,}'),
            ("Forecast horizons", "H1 – H7"),
            ("Models compared", str(len(MODEL_NAMES))),
            ("Best overall model", contracts["best_model"]),
        ]
        for label, value in facts:
            st.markdown(f'<div class="sb-fact"><span>{label}</span><span>{value}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-section-title">About</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="sb-badge">No retraining · saved evidence only</span>',
            unsafe_allow_html=True,
        )

    return selection


# --------------------------------------------------------------------------- #
# Overview KPIs
# --------------------------------------------------------------------------- #
def overview_kpis(data: dict, contracts: dict) -> None:
    ranking = data["ranking"].sort_values("Rank")
    winner = ranking.iloc[0]

    cards = [
        {
            "label": "Best overall model",
            "value": contracts["best_model"],
            "note": "Lowest saved pooled Price RMSE across H1–H7",
            "icon": "🏆",
            "accent": "#f59e0b",
        },
        {
            "label": "Overall Price RMSE",
            "value": f'{winner["Overall_Price_RMSE"]:,.2f}',
            "note": "Average error magnitude with larger errors weighted more",
            "icon": "📉",
            "accent": "#2563eb",
        },
        {
            "label": "RMSE skill vs persistence",
            "value": f'{winner["Overall_RMSE_Skill_vs_Persistence"] * 100:.2f}%',
            "note": "Positive skill means the model beat persistence",
            "icon": "🎯",
            "accent": "#059669",
        },
    ]
    columns = st.columns(3)
    for index, (column, card) in enumerate(zip(columns, cards)):
        column.markdown(
            f"""
            <div class="overview-kpi kpi-delay-{index}" style="--kpi-accent:{card['accent']}">
              <div class="kpi-top">
                <div class="kpi-label">{card['label']}</div>
                <div class="kpi-icon">{card['icon']}</div>
              </div>
              <div class="kpi-value">{card['value']}</div>
              <div class="kpi-note">{card['note']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="validation-ribbon">
          <span class="validation-dot"></span>
          <span>Saved evidence validated</span>
          <span class="validation-separator">•</span><span>4 models</span>
          <span class="validation-separator">•</span><span>7 direct Pipelines each</span>
          <span class="validation-separator">•</span><span>22 ordered predictors</span>
          <span class="validation-separator">•</span><span>No retraining</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def historical_tab(data: dict, contracts: dict, selection: str) -> None:
    page_title("Historical Evaluation results", "Replay a saved walk-forward Evaluation date — no retraining, saved evidence only.")
    date_labels = [date.strftime("%Y-%m-%d") for date in contracts["common_dates"]]
    chosen_label = st.selectbox("Evaluation Origin Date (609 common dates only)", date_labels)
    chosen_date = pd.Timestamp(chosen_label)
    canonical_row = data["canonical"].loc[data["canonical"]["Origin_Date"].eq(chosen_date)].iloc[0]
    with st.expander("Show the Origin Date market inputs", expanded=False):
        cols = st.columns(4)
        for index, field in enumerate(VISIBLE_FIELDS):
            cols[index % 4].number_input(field, value=float(canonical_row[field]), disabled=True, key=f"history_{field}")

    if st.button("Predict from saved Evaluation evidence", type="primary", key="history_predict"):
        detailed_models = selected_models(selection, MODEL_NAMES)
        featured_model = contracts["best_model"] if selection == "All Models" else selection
        featured = historical_results(data["predictions"], chosen_date, [featured_model])
        detailed = historical_results(data["predictions"], chosen_date, detailed_models)
        st.session_state["historical_output"] = (chosen_label, selection, featured, detailed)

    output = st.session_state.get("historical_output")
    if output and output[0] == chosen_label and output[1] == selection:
        _, _, featured, detailed = output
        featured_model = contracts["best_model"] if selection == "All Models" else selection
        show_featured(featured, featured_model, contracts["best_model"])
        section_header("A", "Detailed forecasts", "The results follow the model selected in the sidebar. Show one horizon or all seven horizons.")
        show_result_browser(
            detailed,
            historical=True,
            key_prefix=f'historical_{selection.replace(" ", "_").lower()}',
        )
        section_header("B", "Forecast path", "Compare the seven direct forecast horizons with the current price and the revealed historical path.")
        st.plotly_chart(
            forecast_figure(detailed, float(canonical_row["Current_Price"]), featured_model, historical=True),
            width="stretch", theme=None,
        )
        st.markdown(
            '<div class="status">✓ Actual values are shown here because this date is part of the completed Evaluation period.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Choose an Origin Date above, then click **Predict from saved Evaluation evidence** to see results.")


def manual_tab(data: dict, bundles: dict, contracts: dict, selection: str) -> None:
    page_title("Manual Input Forecast", "Enter current market values to generate a fresh H1–H7 forecast from the saved deployment pipelines.")
    defaults, prior, external = latest_manual_defaults(data["canonical"])
    with st.form("manual_form"):
        columns = st.columns(4)
        entered = {}
        for index, field in enumerate(VISIBLE_FIELDS):
            step = 1.0 if field != "Current_Volume" else 100.0
            entered[field] = columns[index % 4].number_input(field, value=float(defaults[field]), step=step, format="%.6f")

        st.markdown(
            f"""
            <div class="note">
              <div class="note-title">🔒 Read-only stored external values</div>
              <div class="note-grid">
                <div class="note-item"><div class="note-label">USD Index return, lag 1</div><div class="note-value">{external[EXTERNAL_FIELDS[0]]:.6f}</div></div>
                <div class="note-item"><div class="note-label">US 10Y real-yield change, lag 1</div><div class="note-value">{external[EXTERNAL_FIELDS[1]]:.6f}</div></div>
              </div>
              <div class="note-foot">Stored coursework values used by every model — not live market data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("Generate direct H1–H7 forecast", type="primary")

    if submitted:
        errors = validate_manual_inputs(entered)
        if errors:
            for error in errors:
                st.error(error)
        else:
            feature_row = build_feature_row(entered, prior, external, PREDICTORS)
            detailed_models = selected_models(selection, MODEL_NAMES)
            all_forecasts = predict_manual(feature_row, bundles, MODEL_NAMES)
            featured_model = contracts["best_model"] if selection == "All Models" else selection
            featured = all_forecasts.loc[all_forecasts["Model"].eq(featured_model)].copy()
            detailed = all_forecasts.loc[all_forecasts["Model"].isin(detailed_models)].copy()
            signature = tuple(entered.values())
            st.session_state["manual_output"] = (
                selection, signature, featured, detailed, feature_row, all_forecasts,
            )

    output = st.session_state.get("manual_output")
    current_signature = tuple(entered.values())
    if output and output[0] == selection and output[1] == current_signature:
        _, _, featured, detailed, feature_row, _ = output
        featured_model = contracts["best_model"] if selection == "All Models" else selection
        show_featured(featured, featured_model, contracts["best_model"])
        section_header("A", "Detailed forecasts", "The results follow the model selected in the sidebar. Show one horizon or all seven horizons.")
        show_result_browser(
            detailed,
            historical=False,
            key_prefix=f'manual_{selection.replace(" ", "_").lower()}',
        )
        section_header("B", "Forecast path", "See the projected price path across all seven forecast horizons in one full-width chart.")
        st.plotly_chart(
            forecast_figure(
                detailed,
                float(feature_row.iloc[0]["Current_Price"]),
                featured_model,
                historical=False,
            ),
            width="stretch", theme=None,
        )
        with st.expander("View the 22 ordered predictors sent to every Pipeline"):
            show_plain_table(feature_row, {column: "{:.8g}" for column in feature_row.columns})
    else:
        st.info("Fill in the market values above, then click **Generate direct H1–H7 forecast** to see results.")


def comparison_section(data: dict, best_model: str, bundles: dict) -> None:
    st.markdown(
        """
        <div class="comparison-banner">
          <div class="comparison-eyebrow">Model Performance</div>
          <div class="comparison-title">Historical Four-Model Comparison</div>
          <div class="comparison-copy">Compare Ridge, KNN, SVR and XGBoost across the same Evaluation period.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("01", "Overall model ranking", "Models are ranked using the saved pooled Overall Price RMSE across H1–H7.")
    ranking = data["ranking"].sort_values("Rank")
    show_plain_table(ranking, {
        "Overall_Price_RMSE": "{:,.2f}", "Overall_Price_MAE": "{:,.2f}",
        "Overall_Price_MAPE_Percent": "{:.3f}%", "Overall_Return_R2": "{:.4f}",
        "Overall_RMSE_Skill_vs_Persistence": "{:.4f}",
    })
    winner = ranking.iloc[0]
    st.markdown(
        f'<div class="status">★ Best overall model: <b>{best_model}</b> &nbsp;·&nbsp; '
        f'Overall Price RMSE: <b>{winner["Overall_Price_RMSE"]:,.2f}</b> &nbsp;·&nbsp; '
        f'RMSE skill: <b>{winner["Overall_RMSE_Skill_vs_Persistence"] * 100:.2f}%</b></div>',
        unsafe_allow_html=True,
    )

    metrics = data["comparison_metrics"]
    section_header("02", "Price RMSE by horizon", "Compare prediction error for every model from H1 to H7. Lower RMSE indicates better price accuracy.")
    st.plotly_chart(rmse_figure(metrics), width="stretch", theme=None)

    section_header("03", "Explore another metric", "Choose one additional measure below. Its chart is displayed on a separate full-width row for easier reading.")
    metric_options = {
        "RMSE skill vs persistence": "RMSE_Skill_vs_Persistence",
        "Price MAE": "Price_MAE",
        "Price MAPE (%)": "Price_MAPE_Percent",
        "Return R²": "Return_R2",
        "Directional accuracy (%)": "Directional_Accuracy_Percent",
    }
    label = st.selectbox("Additional metric", list(metric_options), key="comparison_metric")
    st.plotly_chart(
        metric_by_horizon_figure(metrics, metric_options[label], label),
        width="stretch", theme=None,
    )

    section_header(
        "04", "Directional accuracy versus Always-Up",
        "Select one horizon for a focused four-model comparison, or choose All Horizons to inspect the complete table.",
    )
    direction_horizon = st.selectbox(
        "Directional accuracy horizon",
        ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "All Horizons"],
        key="direction_accuracy_horizon",
    )
    direction = metrics.loc[
        metrics["Horizon"].ne("Overall"),
        ["Model", "Horizon", "Directional_Accuracy_Percent", "Always_Up_Accuracy_Percent"],
    ].copy()
    if direction_horizon != "All Horizons":
        direction = direction.loc[direction["Horizon"].eq(direction_horizon)].copy()
    show_plain_table(direction, {
        "Directional_Accuracy_Percent": "{:.2f}%", "Always_Up_Accuracy_Percent": "{:.2f}%",
    })

    section_header("05", "Historical H7 performance and future forecast", "The graph shows H7 historical performance and automatically continues beyond the final dataset record with the selected model's future H1–H7 forecast.")
    historical_model = st.selectbox(
        "Historical chart model",
        ["All Models", *MODEL_NAMES],
        index=MODEL_NAMES.index(best_model) + 1,
    )
    historical_horizon = "H7"

    # Always provide a seven-horizon extension using the latest stored input
    # values. A submitted Manual Input forecast replaces this automatic default.
    defaults, prior, external = latest_manual_defaults(data["canonical"])
    latest_feature_row = build_feature_row(defaults, prior, external, PREDICTORS)
    automatic_forecasts = predict_manual(latest_feature_row, bundles, MODEL_NAMES)
    forecast_origin_date = max(
        frame["Target_Date"].max() for frame in data["predictions"].values()
    )
    all_future_forecasts = automatic_forecasts
    forecast_source = "latest stored input values"

    manual_output = st.session_state.get("manual_output")
    if manual_output and len(manual_output) == 6:
        _, _, _, _, _, all_future_forecasts = manual_output
        forecast_source = "latest Manual Input"

    if historical_model == "All Models":
        historical_predictions = pd.concat(
            [data["predictions"][name] for name in MODEL_NAMES],
            ignore_index=True,
        )
        future_results = all_future_forecasts.copy()
    else:
        historical_predictions = data["predictions"][historical_model]
        future_results = all_future_forecasts.loc[
            all_future_forecasts["Model"].eq(historical_model)
        ].copy()
    st.plotly_chart(
        actual_vs_predicted_figure(
            historical_predictions,
            historical_model,
            historical_horizon,
            future_results=future_results,
            forecast_origin_date=forecast_origin_date,
        ),
        width="stretch", theme=None,
    )
    st.caption(
        f'Each model prediction line continues into the {forecast_source} H1–H7 forecast using the same colour and style. '
        'Future positions use business-day spacing; horizons represent recorded observations ahead.'
    )


def main() -> None:
    apply_style()
    st.markdown(
        """
        <div class="hero">
          <div class="hero-content">
            <div class="hero-badge">BMDS2003 · Data Science Project</div>
            <div class="hero-title">Daily Gold Price<br><span class="gold">Forecasting Dashboard</span></div>
            <div class="hero-subtitle">
              Compare four machine-learning models and explore direct H1–H7 gold-price forecasts, powered entirely by saved coursework evidence.
            </div>
            <div class="hero-tags">
              <span class="hero-tag">H1–H7 Direct Forecasts</span>
              <span class="hero-tag">Ridge · KNN · SVR · XGBoost</span>
              <span class="hero-tag">No Retraining</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        data = cached_data()
        bundles = cached_bundles()
        contracts = validate_contracts(data, bundles)
    except Exception as error:
        st.error(f"Startup validation failed: {error}")
        st.stop()

    selection = sidebar(data, contracts)

    overview_kpis(data, contracts)
    st.write("")

    tab1, tab2 = st.tabs(["📅  Existing Evaluation Date", "✍️  Manual Input"])
    with tab1:
        historical_tab(data, contracts, selection)
        comparison_section(data, contracts["best_model"], bundles)
    with tab2:
        manual_tab(data, bundles, contracts, selection)

    with st.expander("📋 Methodology and limitations", expanded=False):
        st.markdown(
            """
            - **Objective:** educational comparison of Ridge, KNN, SVR and XGBoost for direct cumulative gold-price return forecasting.
            - **Direct horizons:** H1–H7 mean recorded observations ahead, not guaranteed calendar days. No H1 forecast is fed recursively into H2–H7.
            - **Leakage safety:** Existing Date mode only replays saved walk-forward Evaluation predictions. Manual mode only calls saved fitted deployment Pipelines. The app never fits, tunes, resplits or shuffles.
            - **Why the two modes may differ:** Historical mode uses the model state available at each Evaluation origin, whereas Manual mode uses the final deployment Pipelines fitted after Evaluation was completed. Both are valid for their different purposes.
            - **Feature construction:** the 15 backend predictors use verified percentage-change, rolling-mean, sample-volatility (`ddof=1`) and momentum formulas. External lag values are the latest stored coursework values.
            - **Limitations:** this is an educational forecasting prototype, not financial advice. No live data is fetched. Regime shift can weaken deployment performance; all predictions are uncertain and are not guaranteed prices.
            """
        )

    st.markdown(
        f"""
        <div class="app-footer">
          <div>Daily Gold Price Forecasting Dashboard · BMDS2003 Data Science Project</div>
          <div>Best overall model: {contracts["best_model"]} · Built with Streamlit &amp; Plotly</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
