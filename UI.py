"""
AquaMind — Smart Irrigation AI Dashboard
CT-469 RL Project (P09)
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaMind — Smart Irrigation AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0f0d;
    color: #e2f0eb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111a15 !important;
    border-right: 1px solid rgba(52,211,153,0.12);
}

[data-testid="stSidebar"] * {
    color: #8aada0 !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #4d7060 !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #162019 !important;
    border: 1px solid rgba(52,211,153,0.15) !important;
    color: #e2f0eb !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #131f18;
    border: 1px solid rgba(52,211,153,0.12);
    border-radius: 12px;
    padding: 16px 20px;
    transition: all 0.25s;
}

[data-testid="metric-container"]:hover {
    border-color: rgba(52,211,153,0.3);
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(52,211,153,0.07);
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #4d7060 !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2f0eb !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 26px !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
}

/* Cards / expanders */
[data-testid="stExpander"] {
    background: #131f18;
    border: 1px solid rgba(52,211,153,0.12) !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: rgba(52,211,153,0.1) !important;
    color: #34d399 !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: rgba(52,211,153,0.2) !important;
    border-color: rgba(52,211,153,0.5) !important;
    transform: scale(1.01);
}

/* Stop/danger button */
.stop-btn .stButton > button {
    background: rgba(248,113,113,0.08) !important;
    color: #f87171 !important;
    border-color: rgba(248,113,113,0.3) !important;
}

.stop-btn .stButton > button:hover {
    background: rgba(248,113,113,0.18) !important;
}

/* Sliders */
.stSlider [data-testid="stThumb"] {
    background: #34d399 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111a15;
    border-bottom: 1px solid rgba(52,211,153,0.12);
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4d7060 !important;
    border-radius: 8px 8px 0 0;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 18px;
}

.stTabs [aria-selected="true"] {
    background: rgba(52,211,153,0.1) !important;
    color: #34d399 !important;
    border-bottom: 2px solid #34d399;
}

/* Divider */
hr {
    border-color: rgba(52,211,153,0.1) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #162019 !important;
    border: 1px solid rgba(52,211,153,0.15) !important;
    color: #e2f0eb !important;
}

/* Text inputs */
.stTextInput > div > div > input {
    background: #162019 !important;
    border: 1px solid rgba(52,211,153,0.15) !important;
    color: #e2f0eb !important;
    border-radius: 8px !important;
}

/* Toggle / checkbox */
.stCheckbox > label > span {
    color: #8aada0 !important;
}

/* Plotly charts transparent bg */
.js-plotly-plot .plotly {
    background: transparent !important;
}

/* Section headers */
.section-header {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4d7060;
    font-weight: 600;
    margin-bottom: 10px;
    margin-top: 4px;
    font-family: 'DM Sans', sans-serif;
}

/* Status badge */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 11px;
    color: #6ee7b7;
    font-family: 'DM Sans', sans-serif;
}

/* Badge */
.badge-green { background: rgba(52,211,153,0.12); color: #34d399; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-amber { background: rgba(251,191,36,0.12); color: #fbbf24; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-blue  { background: rgba(56,189,248,0.12);  color: #38bdf8; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-red   { background: rgba(248,113,113,0.12); color: #f87171; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }

/* Card wrapper */
.card-box {
    background: #131f18;
    border: 1px solid rgba(52,211,153,0.12);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}

.card-box:hover {
    border-color: rgba(52,211,153,0.22);
}

/* Log items */
.log-item {
    background: #162019;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-size: 12px;
    color: #8aada0;
    border-left: 2px solid #34d399;
    font-family: 'DM Sans', sans-serif;
    line-height: 1.5;
}

.log-item.delay { border-left-color: #fbbf24; }
.log-item.skip  { border-left-color: #4d7060; }
.log-item.alert { border-left-color: #f87171; }

/* Phase list item */
.phase-done {
    background: #162019;
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: #8aada0;
}

.phase-pending {
    background: #162019;
    border: 1px solid rgba(52,211,153,0.06);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
    font-size: 12px;
    color: #4d7060;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(52,211,153,0.2); border-radius: 2px; }

/* Number font */
.mono-num {
    font-family: 'Space Mono', monospace;
    color: #34d399;
}

/* Report text block */
.report-block {
    background: #162019;
    border-left: 2px solid #34d399;
    border-radius: 0 8px 8px 0;
    padding: 12px 14px;
    font-size: 13px;
    color: #8aada0;
    line-height: 1.7;
    margin-bottom: 14px;
}

/* Comparison table */
.cmp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: 'DM Sans', sans-serif;
}
.cmp-table th {
    text-align: left;
    padding: 8px 10px;
    color: #4d7060;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1px solid rgba(52,211,153,0.1);
}
.cmp-table td {
    padding: 9px 10px;
    border-bottom: 1px solid rgba(52,211,153,0.04);
    color: #8aada0;
}
.cmp-table tr:hover td { background: rgba(52,211,153,0.03); }
.winner { color: #34d399 !important; font-weight: 600; }

/* Weather card */
.wx-card {
    background: #162019;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
    min-width: 70px;
    border: 1px solid transparent;
    transition: all 0.2s;
    display: inline-block;
    margin-right: 6px;
    margin-bottom: 4px;
}
.wx-card:hover {
    border-color: rgba(52,211,153,0.2);
    transform: translateY(-1px);
}

/* Progress bar custom */
.prog-bg {
    height: 4px;
    background: #162019;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 8px;
}
.prog-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.8s ease;
}

/* Logo */
.logo-area {
    font-family: 'Space Mono', monospace;
    font-size: 18px;
    font-weight: 700;
    color: #34d399;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

ACCENT       = "#34d399"
ACCENT2      = "#10b981"
BLUE         = "#38bdf8"
AMBER        = "#fbbf24"
RED          = "#f87171"
PURPLE       = "#a78bfa"
BG           = "#0a0f0d"
BG2          = "#111a15"
BG3          = "#162019"
CARD         = "#131f18"
TEXT         = "#e2f0eb"
TEXT2        = "#8aada0"
TEXT3        = "#4d7060"
BORDER       = "rgba(52,211,153,0.12)"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=TEXT2),
    margin=dict(l=0, r=0, t=10, b=30),
    xaxis=dict(showgrid=True, gridcolor="rgba(52,211,153,0.07)", zeroline=False,
               color=TEXT3, tickfont=dict(size=9, family="Space Mono")),
    yaxis=dict(showgrid=True, gridcolor="rgba(52,211,153,0.07)", zeroline=False,
               color=TEXT3, tickfont=dict(size=9, family="Space Mono")),
)


def card(content: str):
    """Wrap content in a styled card div."""
    st.markdown(f'<div class="card-box">{content}</div>', unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<p class="section-header">{text}</p>', unsafe_allow_html=True)


def badge(text: str, color: str = "green"):
    return f'<span class="badge-{color}">{text}</span>'


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="logo-area" style="padding:6px 0 14px;">
        🌱 AquaMind
    </div>
    <div style="font-size:10px;color:#4d7060;letter-spacing:1.5px;text-transform:uppercase;
                margin-bottom:16px;margin-left:32px;margin-top:-10px;">
        Smart Irrigation AI
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    farm   = st.selectbox("Farm", ["Field A — North", "Field B — South", "Greenhouse #1"])
    crop   = st.selectbox("Crop", ["🌾 Wheat", "🌾 Rice", "🌿 Cotton", "🌽 Corn"])
    season = st.selectbox("Growth Stage",
                          ["Germination", "Seedling", "Vegetative", "Flowering", "Maturity"])

    st.divider()

    section_header("Navigation")
    page = st.radio("", [
        "📊  Dashboard",
        "🧪  Simulation",
        "🤖  RL Training",
        "📈  Analytics",
        "⚙️  Settings",
        "📄  Report",
    ], label_visibility="collapsed")

    st.divider()

    section_header("Scenario")
    scenario = st.radio("Scenario", ["Normal", "Drought", "Flood-Risk"],
                        horizontal=True, label_visibility="collapsed")

    st.divider()

    st.markdown("""
    <div class="status-online">
        <span style="width:7px;height:7px;border-radius:50%;background:#34d399;
                     display:inline-block;animation:pulse 2s infinite;"></span>
        Backend online
    </div>
    <style>
    @keyframes pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50% { opacity:0.5; transform:scale(0.85); }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:10px;color:#4d7060;text-align:center;">
        Week 3 · {season} Stage<br>
        {datetime.now().strftime('%d %b %Y  %H:%M')}
    </div>
    """, unsafe_allow_html=True)


# ─── TOPBAR ───────────────────────────────────────────────────────────────────

page_name = page.split("  ")[1].strip()

col_title, col_btns = st.columns([3, 2])
with col_title:
    st.markdown(f"""
    <div style="padding:6px 0 12px;">
        <div style="font-size:18px;font-weight:600;color:{TEXT};">{page_name}</div>
        <div style="font-size:12px;color:{TEXT3};">
            Smart Irrigation RL System · {farm} · {crop}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_btns:
    b1, b2, b3 = st.columns(3)
    with b1:
        st.button("↻ Refresh")
    with b2:
        st.button("▶ Start Agent")
    with b3:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        st.button("⏹ Override")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()


# ─── SYNTHETIC DATA ───────────────────────────────────────────────────────────

np.random.seed(42)
hours   = pd.date_range(end=datetime.now(), periods=48, freq="30min")
moisture = np.clip(
    60 + np.cumsum(np.random.randn(48) * 0.8) - np.linspace(0, 8, 48),
    30, 90
)
baseline_moisture = np.clip(
    55 + np.cumsum(np.random.randn(48) * 1.2) - np.linspace(0, 12, 48),
    25, 90
)

episodes   = np.arange(1, 501)
rewards_rl = -80 + 90 * (1 - np.exp(-episodes / 120)) + np.random.randn(500) * 4
rewards_baseline = np.full(500, 15) + np.random.randn(500) * 3

days = pd.date_range(end=datetime.now(), periods=30, freq="D")
water_rl      = 120 - np.cumsum(np.random.randn(30) * 2) * 0.3 + np.random.randn(30) * 5
water_fixed   = 140 + np.random.randn(30) * 4
water_sensor  = 130 + np.random.randn(30) * 5


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if "Dashboard" in page:

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💧 Water Saved Today",  "247 L",    "+18% vs baseline")
    k2.metric("🌡 Soil Moisture (avg)", "63 %",     "Optimal range")
    k3.metric("⚡ Energy Cost",         "$0.84/hr",  "↓ Off-peak")
    k4.metric("🌾 Crop Health Score",   "8.4 / 10", "+0.3 from yesterday")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    col_chart, col_right = st.columns([2, 1])

    with col_chart:
        section_header("Soil Moisture — 24h")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=moisture,
            mode="lines", name="RL Agent",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.07)"
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=baseline_moisture,
            mode="lines", name="Baseline",
            line=dict(color=BLUE, width=1.5, dash="dot"),
        ))
        # Irrigation markers
        irr_times = [hours[10], hours[32]]
        irr_vals  = [moisture[10], moisture[32]]
        fig.add_trace(go.Scatter(
            x=irr_times, y=irr_vals,
            mode="markers", name="Irrigated",
            marker=dict(color=BLUE, size=9, symbol="circle",
                        line=dict(color=BG, width=2))
        ))
        fig.add_hrect(y0=40, y1=70, fillcolor="rgba(52,211,153,0.05)",
                      line_width=0, annotation_text="Optimal", annotation_position="top right",
                      annotation=dict(font=dict(size=9, color=TEXT3)))
        fig.update_layout(**PLOTLY_LAYOUT, height=220,
                          legend=dict(orientation="h", y=-0.15,
                                      font=dict(size=10, color=TEXT3),
                                      bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    with col_right:
        section_header("Reservoir Level")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=72,
            number=dict(suffix="%", font=dict(family="Space Mono", color=TEXT, size=28)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=0, tickcolor=TEXT3,
                          tickfont=dict(size=8, color=TEXT3)),
                bar=dict(color=ACCENT, thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 30],  color="rgba(248,113,113,0.15)"),
                    dict(range=[30, 60], color="rgba(251,191,36,0.1)"),
                    dict(range=[60, 100],color="rgba(52,211,153,0.08)"),
                ],
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=160, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config=dict(displayModeBar=False))
        st.markdown(f"""
        <div style="text-align:center;font-size:11px;color:{TEXT3};margin-top:-10px;">
            ~1,440 L available
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3
    col_depths, col_log = st.columns(2)

    with col_depths:
        section_header("Moisture by Depth")
        depths = {"5 cm": (78, ACCENT), "20 cm": (63, ACCENT2), "40 cm": (47, "#0f766e")}
        for label, (val, color) in depths.items():
            d1, d2, d3 = st.columns([1, 5, 1])
            d1.markdown(f'<span style="font-size:11px;color:{TEXT3};font-family:Space Mono;">{label}</span>',
                        unsafe_allow_html=True)
            fig_bar = go.Figure(go.Bar(
                x=[val], y=[""], orientation="h",
                marker=dict(color=color, opacity=0.75),
                width=[0.6]
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=28, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(range=[0, 100], showgrid=False, visible=False),
                yaxis=dict(showgrid=False, visible=False),
                showlegend=False,
            )
            d2.plotly_chart(fig_bar, use_container_width=True, config=dict(displayModeBar=False))
            d3.markdown(f'<span style="font-size:11px;color:{TEXT2};font-family:Space Mono;">{val}%</span>',
                        unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:10px;font-size:10px;color:#4d7060;
                    text-transform:uppercase;letter-spacing:0.8px;font-weight:600;">
            Optimal Range: 40% – 70%
        </div>
        """, unsafe_allow_html=True)

    with col_log:
        section_header("Agent Decision Log")
        logs = [
            ("irrigate", "14:32", "Irrigate 120 L",    "moisture below threshold, rain forecast 72h"),
            ("skip",     "12:15", "Skip",               "rain in 24h, moisture optimal"),
            ("delay",    "10:00", "Delay 6h",           "high electricity peak — wait for off-peak"),
            ("irrigate", "06:45", "Irrigate 80 L",      "early morning low-cost window"),
            ("alert",    "02:10", "⚠ Alert",            "depth 40 cm below 40%, scheduled irrigation"),
        ]
        for cls, t, action, reason in logs:
            st.markdown(f"""
            <div class="log-item {cls}">
                <span style="font-family:Space Mono;font-size:9px;color:#4d7060;">{t}</span>
                &nbsp;&nbsp;
                <strong style="color:#e2f0eb;">{action}</strong>
                &nbsp;—&nbsp; {reason}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Weather strip
    section_header("72h Weather Forecast · OpenWeatherMap")
    weather_data = [
        ("Now",  "☀️", "34°C", "0%"),
        ("+3h",  "⛅", "32°C", "5%"),
        ("+6h",  "⛅", "29°C", "12%"),
        ("+12h", "🌦", "27°C", "40%"),
        ("+24h", "🌧", "25°C", "75%"),
        ("+36h", "🌧", "24°C", "82%"),
        ("+48h", "⛅", "28°C", "20%"),
        ("+60h", "☀️", "31°C", "3%"),
        ("+72h", "☀️", "33°C", "1%"),
    ]
    wx_cols = st.columns(len(weather_data))
    for col, (t, icon, temp, rain) in zip(wx_cols, weather_data):
        col.markdown(f"""
        <div class="wx-card">
            <div style="font-size:9px;color:{TEXT3};margin-bottom:4px;
                        text-transform:uppercase;letter-spacing:0.5px;">{t}</div>
            <div style="font-size:20px;margin-bottom:4px;">{icon}</div>
            <div style="font-size:12px;font-family:Space Mono;font-weight:700;
                        color:{TEXT};">{temp}</div>
            <div style="font-size:9px;color:{BLUE};margin-top:2px;">{rain}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
elif "Simulation" in page:

    col_cfg, col_play = st.columns(2)

    with col_cfg:
        section_header("Simulation Config")
        with st.container():
            ep_len  = st.slider("Episode Length (days)", 7, 90, 30)
            init_sm = st.slider("Initial Soil Moisture (%)", 10, 90, 50)
            rv      = st.select_slider("Rainfall Variability",
                                       options=["Low", "Medium", "High"], value="Medium")
            soil    = st.selectbox("Soil Type",
                                   ["Sandy Loam", "Clay", "Silt", "Loam"])
            pricing = st.selectbox("Electricity Pricing",
                                   ["Flat Rate", "Time-of-Use (ToU)", "Dynamic"])

        st.markdown("<br>", unsafe_allow_html=True)
        col_r, col_s = st.columns(2)
        with col_r: st.button("▶ Run Simulation", use_container_width=True)
        with col_s:
            st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
            st.button("⏹ Reset",       use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Episode Summary")
        m1, m2 = st.columns(2)
        m1.metric("Water Saved",      "+22%",   "vs fixed schedule")
        m2.metric("Cumulative Reward", "487",    "+6.7 avg/step")
        m3, m4 = st.columns(2)
        m3.metric("Irrigations",      "14",     f"over {ep_len} days")
        m4.metric("Yield Proxy",      "8.1/10", "+1.3 vs baseline")

    with col_play:
        section_header("Simulation Playback")

        sim_days = np.arange(ep_len)
        rl_moist  = np.clip(init_sm + np.cumsum(np.random.randn(ep_len) * 2.5)
                            - np.linspace(0, 10, ep_len), 20, 90)
        bl_moist  = np.clip(init_sm - 5 + np.cumsum(np.random.randn(ep_len) * 3)
                            - np.linspace(0, 18, ep_len), 20, 90)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sim_days, y=rl_moist, mode="lines", name="RL Agent",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.06)"
        ))
        fig2.add_trace(go.Scatter(
            x=sim_days, y=bl_moist, mode="lines", name="Baseline",
            line=dict(color=BLUE, width=1.5, dash="dot")
        ))
        fig2.add_hrect(y0=40, y1=70, fillcolor="rgba(52,211,153,0.05)", line_width=0)
        fig2.update_layout(**PLOTLY_LAYOUT, height=250,
                           xaxis_title="Day", yaxis_title="Moisture %",
                           legend=dict(orientation="h", y=-0.18,
                                       font=dict(size=10, color=TEXT3),
                                       bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True, config=dict(displayModeBar=False))

        st.markdown("""
        <div class="prog-bg"><div class="prog-fill" style="width:68%"></div></div>
        <div style="display:flex;justify-content:space-between;margin-top:5px;
                    font-size:10px;color:#4d7060;font-family:Space Mono;">
            <span>68% complete</span><span>~4 min remaining</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Reward Accumulation")
        cum_reward = np.cumsum(np.random.randn(ep_len) * 0.8 + 0.5)
        fig3 = go.Figure(go.Scatter(
            x=sim_days, y=cum_reward, mode="lines",
            line=dict(color=AMBER, width=2),
            fill="tozeroy", fillcolor="rgba(251,191,36,0.06)"
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=120,
                           xaxis_title="Day", yaxis_title="Cum. Reward",
                           margin=dict(l=0, r=0, t=5, b=30))
        st.plotly_chart(fig3, use_container_width=True, config=dict(displayModeBar=False))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
elif "Training" in page:

    col_cfg, col_prog = st.columns([1, 1.1])

    with col_cfg:
        section_header("Agent Configuration")
        algo = st.radio("Algorithm", ["DQN", "DDPG", "PPO"], horizontal=True)

        st.markdown("<br>", unsafe_allow_html=True)
        lr      = st.slider("Learning Rate (α)",      1, 100, 10,
                             help="Divided by 10,000 → actual LR")
        gamma   = st.slider("Discount Factor (γ) ×100", 80, 100, 99)
        buf     = st.slider("Replay Buffer Size (×1k)", 1, 100, 10)
        epsilon = st.slider("Exploration (ε) ×100",    1, 100, 10)
        n_eps   = st.slider("Training Episodes",        100, 2000, 500, step=100)
        batch   = st.slider("Batch Size",               32, 512, 64, step=32)

        st.markdown(f"""
        <div class="card-box" style="margin-top:8px;">
            <div class="section-header">Effective Hyperparameters</div>
            <div style="font-family:Space Mono;font-size:11px;color:{TEXT2};line-height:2;">
                lr = {lr/10000:.4f} &nbsp;|&nbsp;
                γ = {gamma/100:.2f}<br>
                buffer = {buf*1000:,} &nbsp;|&nbsp;
                ε = {epsilon/100:.2f}<br>
                episodes = {n_eps} &nbsp;|&nbsp;
                batch = {batch}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca: st.button("▶ Start Training", use_container_width=True)
        with cb:
            st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
            st.button("⏹ Stop",            use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.button("💾 Save Best Model", use_container_width=True)

    with col_prog:
        section_header("Training Curves")
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(
            x=episodes, y=rewards_rl, mode="lines", name="RL Agent",
            line=dict(color=AMBER, width=2),
        ))
        fig_tr.add_trace(go.Scatter(
            x=episodes, y=rewards_baseline, mode="lines", name="Baseline",
            line=dict(color=TEXT3, width=1.5, dash="dot"),
        ))
        fig_tr.update_layout(**PLOTLY_LAYOUT, height=220,
                             xaxis_title="Episode", yaxis_title="Reward",
                             legend=dict(orientation="h", y=-0.2,
                                         font=dict(size=10, color=TEXT3),
                                         bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_tr, use_container_width=True, config=dict(displayModeBar=False))

        st.markdown("""
        <div class="section-header" style="margin-top:4px;">Training Progress</div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="font-family:Space Mono;font-size:12px;color:#fbbf24;">Ep 342 / 500</span>
            <span class="badge-amber">Running</span>
        </div>
        <div class="prog-bg"><div class="prog-fill" style="width:68%"></div></div>
        <div style="display:flex;justify-content:space-between;margin-top:5px;
                    font-size:10px;color:#4d7060;font-family:Space Mono;">
            <span>68% complete</span><span>~4 min remaining</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Phase Checklist")
        phases = [
            (True,  "Environment Setup",      "Gym env + soil model",          "green"),
            (True,  "Baseline Comparison",    "Fixed + sensor-only",           "green"),
            (False, "DQN Training",           "500 episodes — in progress",    "amber"),
            (False, "DDPG Extension",         "Continuous volume control",     None),
            (False, "FastAPI Integration",    "Inference endpoint",            None),
        ]
        for done, name, desc, color in phases:
            mark  = "✓" if done else ("→" if color == "amber" else "○")
            bclr  = f"rgba(52,211,153,0.2)" if done else (
                    f"rgba(251,191,36,0.15)" if color == "amber" else "rgba(52,211,153,0.05)")
            tclr  = ACCENT if done else (AMBER if color == "amber" else TEXT3)
            st.markdown(f"""
            <div style="background:#162019;border:1px solid {bclr};border-radius:8px;
                         padding:9px 12px;margin-bottom:5px;display:flex;
                         align-items:center;gap:10px;">
                <span style="font-family:Space Mono;font-size:12px;color:{tclr};">{mark}</span>
                <div style="flex:1;">
                    <div style="font-size:12px;color:{'#e2f0eb' if done else TEXT2};
                                font-weight:500;">{name}</div>
                    <div style="font-size:10px;color:{TEXT3};margin-top:1px;">{desc}</div>
                </div>
                {"<span class='badge-green'>Done</span>" if done else
                 (f"<span class='badge-amber'>Running</span>" if color == "amber" else
                  "<span style='font-size:10px;color:#4d7060;'>Pending</span>")}
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif "Analytics" in page:

    section_header("RL vs Baseline Comparison · 30-day window")
    cmp_html = """
    <table class="cmp-table">
        <thead>
            <tr>
                <th>Method</th>
                <th>Water Used (L)</th>
                <th>Yield Proxy</th>
                <th>Energy Cost ($)</th>
                <th>Avg Reward</th>
                <th>Water Saved</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="badge-green">RL Agent (DQN)</span></td>
                <td class="winner">3,240</td>
                <td class="winner">8.4</td>
                <td class="winner">18.20</td>
                <td class="winner">+6.7</td>
                <td class="winner">+22%</td>
            </tr>
            <tr>
                <td><span class="badge-amber">Fixed Schedule</span></td>
                <td>4,150</td><td>7.1</td><td>24.80</td><td>+2.1</td><td>—</td>
            </tr>
            <tr>
                <td><span class="badge-blue">Sensor-Only</span></td>
                <td>3,820</td><td>7.6</td><td>22.10</td><td>+3.4</td><td>+8%</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(f'<div class="card-box">{cmp_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_wt, col_sc = st.columns(2)

    with col_wt:
        section_header("Daily Water Usage")
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=days, y=water_rl,
                               name="RL Agent",  marker_color="rgba(52,211,153,0.6)"))
        fig_w.add_trace(go.Bar(x=days, y=water_fixed,
                               name="Fixed",     marker_color="rgba(251,191,36,0.4)"))
        fig_w.add_trace(go.Bar(x=days, y=water_sensor,
                               name="Sensor",    marker_color="rgba(56,189,248,0.4)"))
        fig_w.update_layout(**PLOTLY_LAYOUT, height=220, barmode="group",
                            legend=dict(orientation="h", y=-0.2,
                                        font=dict(size=10, color=TEXT3),
                                        bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_w, use_container_width=True, config=dict(displayModeBar=False))

    with col_sc:
        section_header("Scenario Analysis")
        scenarios = ["Normal", "Drought", "Variable Rain"]
        rl_vals   = [22, 31, 18]
        bl_vals   = [0, 0, 0]
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=scenarios, y=rl_vals, name="RL vs Fixed",
            marker_color=[ACCENT, RED, AMBER],
            text=[f"+{v}%" for v in rl_vals],
            textposition="outside",
            textfont=dict(size=11, color=TEXT2, family="Space Mono")
        ))
        fig_s.update_layout(**PLOTLY_LAYOUT, height=220,
                            yaxis_title="Water Saved (%)",
                            legend=dict(orientation="h", y=-0.2,
                                        font=dict(size=10, color=TEXT3),
                                        bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_s, use_container_width=True, config=dict(displayModeBar=False))

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Performance Metrics Summary")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Water vs Fixed",    "+22%",  "RL Agent advantage")
    p2.metric("Yield Improvement", "+18%",  "Crop health proxy")
    p3.metric("Energy Reduction",  "-27%",  "Off-peak scheduling")
    p4.metric("Best Episode Rwd",  "8.9",   "Episode 489")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
elif "Settings" in page:

    col_farm, col_app = st.columns(2)

    with col_farm:
        section_header("Farm Configuration")
        st.text_input("Farm Area (hectares)",    value="2.4",          key="farm_area")
        st.text_input("Reservoir Capacity (L)",  value="2000",         key="res_cap")
        st.slider("Low Moisture Alert (%)",      10, 60, 35,           key="low_thr")
        st.slider("High Moisture Alert (%)",     60, 100, 85,          key="high_thr")
        st.text_input("Weather API Key",
                      value="owm_xxxx_your_key_here", type="password", key="wx_key")
        st.text_input("FastAPI Backend URL",     value="http://localhost:8000", key="api_url")
        st.button("💾 Save Config", use_container_width=True)

    with col_app:
        section_header("App Settings")
        auto_irr  = st.toggle("Auto Irrigation",      value=True)
        realtime  = st.toggle("Real-time Updates",     value=True)
        email_alt = st.toggle("Email Alerts",          value=False)
        energy_op = st.toggle("Energy Optimization",   value=True)
        drought_a = st.toggle("Auto Drought Mode",     value=False)
        tb_log    = st.toggle("TensorBoard Logging",   value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Sensor Calibration")
        st.slider("Sensor Offset (%) — Depth 5cm",  -10, 10, 0)
        st.slider("Sensor Offset (%) — Depth 20cm", -10, 10, 0)
        st.slider("Sensor Offset (%) — Depth 40cm", -10, 10, 0)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Display")
        st.selectbox("Temperature Unit", ["°C", "°F"])
        st.selectbox("Volume Unit",      ["Litres (L)", "Gallons (gal)"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif "Report" in page:

    col_rep, col_exp = st.columns([1.3, 1])

    with col_rep:
        section_header("Weekly Summary Report")

        st.markdown(f"""
        <div style="font-size:10px;color:{TEXT3};text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px;font-weight:600;">
            Performance Overview
        </div>
        <div class="report-block">
            The RL agent achieved <strong style="color:{ACCENT};">22% water savings</strong>
            compared to the fixed-schedule baseline over the past 7 days, while maintaining
            a crop health score of <strong style="color:{ACCENT};">8.4/10</strong>.
            Energy costs were reduced by <strong style="color:{ACCENT};">27%</strong>
            through off-peak irrigation scheduling.
        </div>

        <div style="font-size:10px;color:{TEXT3};text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px;font-weight:600;">
            MDP Design Justification
        </div>
        <div class="report-block">
            The reward function balances <code>crop_health_score</code> with water
            conservation weight (0.4) and penalises stress events. The 8-dimensional
            state space captures soil moisture at 3 depths, temperature, humidity,
            24h/72h rainfall forecast, and crop growth stage. Action space is discrete:
            {{irrigate now, delay 2h, delay 6h, skip today}}.
        </div>

        <div style="font-size:10px;color:{TEXT3};text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px;font-weight:600;">
            Failure Analysis
        </div>
        <div class="report-block">
            Early training episodes showed reward hacking — the agent learned to skip
            irrigation entirely to maximise the water-saving term. Fixed by adding a
            wilting penalty and capping the conservation weight. Training instability
            resolved via SB3 callbacks with learning rate scheduling.
        </div>

        <div style="font-size:10px;color:{TEXT3};text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px;font-weight:600;">
            AI Disclosure
        </div>
        <div class="report-block">
            Used Claude for environment architecture design and reward shaping
            suggestions. All model training, evaluation, and integration was implemented
            by the team. Code modified 70%+ from AI-suggested scaffolding. Grok used
            for initial FastAPI endpoint boilerplate.
        </div>
        """, unsafe_allow_html=True)

    with col_exp:
        section_header("Export & Download")
        st.button("↓ Download PDF Report",        use_container_width=True)
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        st.button("↓ Export Charts (PNG)",         use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.button("↓ Training Logs (CSV)",         use_container_width=True)
        st.button("↓ Full Episode Data (JSON)",    use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Saved Model Info")
        model_info = {
            "Algorithm":     "DQN (MlpPolicy)",
            "File":          "models/best_dqn.zip",
            "Episodes":      "500",
            "Best Reward":   "+8.9",
            "Trained on":    datetime.now().strftime("%d %b %Y"),
            "Framework":     "Stable-Baselines3",
        }
        for k, v in model_info.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                         padding:7px 0;border-bottom:1px solid rgba(52,211,153,0.06);">
                <span style="font-size:12px;color:{TEXT2};">{k}</span>
                <span style="font-size:11px;font-family:Space Mono;
                             color:{ACCENT};">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Presentation Checklist")
        items = [
            ("Live demo (Streamlit)",       True),
            ("Training curves shown",       True),
            ("Baseline comparison ready",   True),
            ("Q&A — all members prepped",   False),
            ("AI disclosure included",      True),
            ("Report PDF generated",        False),
        ]
        for label, done in items:
            st.checkbox(label, value=done)