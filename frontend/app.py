import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Config ───────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title = "Smart Irrigation System",
    page_icon  = "🌾",
    layout     = "wide"
)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("🌾 Smart Irrigation")
st.sidebar.markdown("RL-based irrigation controller for wheat farming — Multan, Pakistan")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Simulation", "Comparison", "Scenario Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Backend status**")

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success("API connected ✓")
    st.sidebar.caption(f"Model loaded: {health['model_loaded']}")
except:
    st.sidebar.error("API not connected ✗")
    st.sidebar.caption("Make sure backend is running on port 8000")

# ════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("🌾 Farm Dashboard")
    st.markdown("Live weather + RL agent recommendation for today")

    # Fetch live weather
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        refresh = st.button("🔄 Refresh")

    try:
        weather = requests.get(f"{API_URL}/weather", timeout=5).json()

        # Weather metrics
        st.subheader("Current Weather — Multan, Pakistan")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Temperature",    f"{weather['temp_c']}°C")
        c2.metric("Humidity",       f"{weather['humidity_pct']}%")
        c3.metric("Current Rain",   f"{weather['rainfall_mm']}mm")
        c4.metric("24h Forecast",   f"{weather['forecast_24h_mm']}mm")
        c5.metric("72h Forecast",   f"{weather['forecast_72h_mm']}mm")

        st.markdown("---")

        # Soil moisture inputs
        st.subheader("Current Soil State")
        st.caption("Adjust sliders to match your sensor readings")

        col1, col2, col3 = st.columns(3)
        with col1:
            m_surface = st.slider("Surface moisture (0-10cm)",  0.0, 0.5, 0.27, 0.01)
        with col2:
            m_root    = st.slider("Root zone moisture (10-30cm)", 0.0, 0.5, 0.25, 0.01)
        with col3:
            m_deep    = st.slider("Deep moisture (30-60cm)",    0.0, 0.5, 0.30, 0.01)

        col4, col5 = st.columns(2)
        with col4:
            crop_stage = st.selectbox(
                "Crop growth stage",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: {
                    0: "0 — Germination",
                    1: "1 — Vegetative",
                    2: "2 — Tillering",
                    3: "3 — Flowering ⚠️",
                    4: "4 — Grain fill",
                    5: "5 — Maturity"
                }[x],
                index=1
            )
        with col5:
            days_progress = st.slider("Season progress", 0.0, 1.0, 0.3, 0.01)

        st.markdown("---")

        # Get RL recommendation
        st.subheader("RL Agent Recommendation")

        payload = {
            "soil_moisture_surface" : m_surface,
            "soil_moisture_root"    : m_root,
            "soil_moisture_deep"    : m_deep,
            "temperature_c"         : weather["temp_c"],
            "humidity_pct"          : weather["humidity_pct"],
            "rainfall_mm"           : weather["rainfall_mm"],
            "forecast_24h_mm"       : weather["forecast_24h_mm"],
            "forecast_72h_mm"       : weather["forecast_72h_mm"],
            "crop_stage"            : crop_stage,
            "days_progress"         : days_progress,
            "electricity_price"     : 0.5,
        }

        pred = requests.post(f"{API_URL}/predict", json=payload, timeout=5).json()

        # Display recommendation
        action = pred["action"]
        if action == 0:
            st.success(f"✅ **{pred['label']}** — {pred['explanation']}")
        else:
            st.warning(f"💧 **{pred['label']}** — {pred['explanation']}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Recommended Action", pred["label"])
        col_b.metric("Water Volume",       f"{pred['volume_litres']}L")
        col_c.metric("Root Zone Moisture", f"{m_root:.2f}")

        # Moisture gauge
        st.markdown("---")
        st.subheader("Soil Moisture Levels")

        fig = go.Figure()
        layers      = ["Surface (0-10cm)", "Root zone (10-30cm)", "Deep (30-60cm)"]
        moistures   = [m_surface, m_root, m_deep]
        bar_colors  = []
        for m in moistures:
            if m < 0.12:
                bar_colors.append("#E24B4A")    # red — wilting
            elif m < 0.20:
                bar_colors.append("#EF9F27")    # amber — dry
            elif m <= 0.35:
                bar_colors.append("#1D9E75")    # green — optimal
            else:
                bar_colors.append("#378ADD")    # blue — waterlogged

        fig.add_trace(go.Bar(
            x      = layers,
            y      = moistures,
            marker_color = bar_colors,
            text   = [f"{m:.2f}" for m in moistures],
            textposition = "auto",
        ))
        fig.add_hline(y=0.12, line_dash="dash", line_color="#E24B4A",
                      annotation_text="Wilting point")
        fig.add_hline(y=0.35, line_dash="dash", line_color="#378ADD",
                      annotation_text="Field capacity")
        fig.update_layout(
            yaxis_range = [0, 0.5],
            yaxis_title = "Moisture fraction",
            height      = 350,
            showlegend  = False,
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not connect to backend: {e}")


# ════════════════════════════════════════════════════════
# PAGE 2 — SIMULATION
# ════════════════════════════════════════════════════════
elif page == "Simulation":
    st.title("🔬 Season Simulation")
    st.markdown("Run a full 120-day wheat season and visualize the results")

    col1, col2, col3 = st.columns(3)
    with col1:
        scenario = st.selectbox("Weather scenario", ["normal", "drought", "wet"])
    with col2:
        agent    = st.selectbox("Agent", ["rl", "fixed", "threshold"],
                                format_func=lambda x: {
                                    "rl"       : "DQN (RL Agent)",
                                    "fixed"    : "Fixed Schedule",
                                    "threshold": "Threshold Sensor"
                                }[x])
    with col3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        run_btn = st.button("▶ Run Simulation", type="primary")

    if run_btn:
        with st.spinner("Running 120-day simulation..."):
            try:
                result = requests.post(
                    f"{API_URL}/simulate",
                    json={"scenario": scenario, "agent": agent, "seed": 42},
                    timeout=60
                ).json()

                # Summary metrics
                st.markdown("---")
                st.subheader("Season Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Water Used", f"{result['total_water_L']:,}L")
                c2.metric("Final Yield",       f"{result['final_yield']}/100")
                c3.metric("Total Reward",      f"{result['total_reward']:.1f}")
                c4.metric("Days Simulated",    f"{len(result['days'])}")

                # Soil moisture chart
                st.markdown("---")
                st.subheader("Root Zone Moisture Over Season")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x    = result["days"],
                    y    = result["moisture"],
                    mode = "lines",
                    name = "Root moisture",
                    line = dict(color="#1D9E75", width=2)
                ))
                fig1.add_hline(y=0.12, line_dash="dash", line_color="#E24B4A",
                               annotation_text="Wilting point (0.12)")
                fig1.add_hline(y=0.35, line_dash="dash", line_color="#378ADD",
                               annotation_text="Field capacity (0.35)")
                fig1.update_layout(
                    xaxis_title = "Day of season",
                    yaxis_title = "Moisture fraction",
                    yaxis_range = [0, 0.5],
                    height      = 350,
                )
                st.plotly_chart(fig1, use_container_width=True)

                # Irrigation actions chart
                st.subheader("Daily Irrigation Actions")
                action_labels = {0: "Skip", 1: "50L", 2: "150L", 3: "300L"}
                action_colors = {0: "#B4B2A9", 1: "#9FE1CB", 2: "#1D9E75", 3: "#0F6E56"}
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x            = result["days"],
                    y            = result["actions"],
                    marker_color = [action_colors[a] for a in result["actions"]],
                    name         = "Action",
                ))
                fig2.update_layout(
                    xaxis_title  = "Day of season",
                    yaxis_title  = "Action",
                    yaxis        = dict(
                        tickvals = [0, 1, 2, 3],
                        ticktext = ["Skip", "50L", "150L", "300L"]
                    ),
                    height       = 300,
                )
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Simulation failed: {e}")


# ════════════════════════════════════════════════════════
# PAGE 3 — COMPARISON
# ════════════════════════════════════════════════════════
elif page == "Comparison":
    st.title("📊 Agent Comparison")
    st.markdown("Compare RL agent vs baseline methods")

    scenario = st.selectbox("Select scenario", ["normal", "drought", "wet"])

    if st.button("▶ Run Comparison", type="primary"):
        with st.spinner("Running all 3 agents..."):
            try:
                results = {}
                for agent in ["rl", "fixed", "threshold"]:
                    r = requests.post(
                        f"{API_URL}/simulate",
                        json={"scenario": scenario, "agent": agent, "seed": 42},
                        timeout=60
                    ).json()
                    results[agent] = r

                st.markdown("---")

                # Summary table
                st.subheader("Results Summary")
                df = pd.DataFrame({
                    "Agent"        : ["Fixed Schedule", "Threshold", "DQN (RL)"],
                    "Water Used (L)": [
                        results["fixed"]["total_water_L"],
                        results["threshold"]["total_water_L"],
                        results["rl"]["total_water_L"]
                    ],
                    "Final Yield"  : [
                        results["fixed"]["final_yield"],
                        results["threshold"]["final_yield"],
                        results["rl"]["final_yield"]
                    ],
                    "Total Reward" : [
                        results["fixed"]["total_reward"],
                        results["threshold"]["total_reward"],
                        results["rl"]["total_reward"]
                    ],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Water savings
                fixed_water = results["fixed"]["total_water_L"]
                rl_water    = results["rl"]["total_water_L"]
                if fixed_water > 0:
                    savings = (fixed_water - rl_water) / fixed_water * 100
                    st.success(f"✅ RL saves **{savings:.1f}%** water vs Fixed Schedule "
                               f"({fixed_water:,}L → {rl_water:,}L)")

                # Bar charts
                col1, col2 = st.columns(2)

                with col1:
                    fig_w = px.bar(
                        df, x="Agent", y="Water Used (L)",
                        color="Agent",
                        color_discrete_sequence=["#D85A30", "#378ADD", "#1D9E75"],
                        title="Water consumption per season"
                    )
                    fig_w.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_w, use_container_width=True)

                with col2:
                    fig_y = px.bar(
                        df, x="Agent", y="Final Yield",
                        color="Agent",
                        color_discrete_sequence=["#D85A30", "#378ADD", "#1D9E75"],
                        title="Crop yield score"
                    )
                    fig_y.update_layout(showlegend=False, height=350, yaxis_range=[0, 110])
                    st.plotly_chart(fig_y, use_container_width=True)

                # Moisture curves
                st.subheader("Soil Moisture Comparison")
                fig_m = go.Figure()
                colors = {"fixed": "#D85A30", "threshold": "#378ADD", "rl": "#1D9E75"}
                names  = {"fixed": "Fixed Schedule", "threshold": "Threshold", "rl": "DQN (RL)"}
                for agent_key in ["fixed", "threshold", "rl"]:
                    fig_m.add_trace(go.Scatter(
                        x    = results[agent_key]["days"],
                        y    = results[agent_key]["moisture"],
                        mode = "lines",
                        name = names[agent_key],
                        line = dict(color=colors[agent_key], width=2)
                    ))
                fig_m.add_hline(y=0.12, line_dash="dash", line_color="red",
                                annotation_text="Wilting point")
                fig_m.add_hline(y=0.35, line_dash="dash", line_color="blue",
                                annotation_text="Field capacity")
                fig_m.update_layout(
                    xaxis_title = "Day of season",
                    yaxis_title = "Root zone moisture",
                    yaxis_range = [0, 0.5],
                    height      = 400,
                )
                st.plotly_chart(fig_m, use_container_width=True)

            except Exception as e:
                st.error(f"Comparison failed: {e}")


# ════════════════════════════════════════════════════════
# PAGE 4 — SCENARIO ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "Scenario Analysis":
    st.title("🌦️ Scenario Analysis")
    st.markdown("Pre-computed results across Normal, Drought, and Wet years")

    try:
        data = requests.get(f"{API_URL}/report/scenario", timeout=5).json()

        scenarios   = ["normal", "drought", "wet"]
        agent_names = ["Fixed Schedule", "Threshold", "DQN (RL)"]
        api_keys    = ["Fixed Schedule", "Threshold", "DQN (RL)"]

        # Build summary dataframe
        rows = []
        for scenario in scenarios:
            for agent in api_keys:
                if scenario in data and agent in data[scenario]:
                    r = data[scenario][agent]
                    rows.append({
                        "Scenario"     : scenario.capitalize(),
                        "Agent"        : agent,
                        "Water (L)"    : r["total_water_L"],
                        "Yield"        : r["final_yield"],
                        "Reward"       : r["total_reward"],
                        "Avg Moisture" : r["avg_moisture"],
                    })

        df = pd.DataFrame(rows)

        # Full table
        st.subheader("Full Results Table")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Water usage across scenarios
        st.subheader("Water Usage Across Scenarios")
        fig_w = px.bar(
            df, x="Scenario", y="Water (L)",
            color="Agent",
            barmode="group",
            color_discrete_sequence=["#D85A30", "#378ADD", "#1D9E75"],
            title="Water consumption — all scenarios"
        )
        fig_w.update_layout(height=400)
        st.plotly_chart(fig_w, use_container_width=True)

        # Yield across scenarios
        st.subheader("Yield Score Across Scenarios")
        fig_y = px.bar(
            df, x="Scenario", y="Yield",
            color="Agent",
            barmode="group",
            color_discrete_sequence=["#D85A30", "#378ADD", "#1D9E75"],
            title="Crop yield — all scenarios"
        )
        fig_y.update_layout(height=400, yaxis_range=[0, 110])
        st.plotly_chart(fig_y, use_container_width=True)

        # Water savings summary
        st.markdown("---")
        st.subheader("RL Water Savings vs Fixed Schedule")
        for scenario in scenarios:
            if scenario in data:
                fixed_w = data[scenario].get("Fixed Schedule", {}).get("total_water_L", 0)
                rl_w    = data[scenario].get("DQN (RL)", {}).get("total_water_L", 0)
                if fixed_w > 0:
                    savings = (fixed_w - rl_w) / fixed_w * 100
                    st.metric(
                        label = f"{scenario.capitalize()} year",
                        value = f"{rl_w:,}L used",
                        delta = f"-{savings:.1f}% vs Fixed ({fixed_w:,}L)"
                    )

    except Exception as e:
        st.error(f"Could not load scenario data: {e}")