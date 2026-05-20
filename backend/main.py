# main.py
import os
import json
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from stable_baselines3 import DQN
from dotenv import load_dotenv

# Local imports
from environment.irrigation_env import IrrigationEnv
from environment.soil_model import SoilModel
from environment.weather_sim import WeatherSimulator
from baselines.baseline_agents import FixedScheduleBaseline, ThresholdBaseline
from data.weather_client import get_live_weather

load_dotenv()

# ── App setup ────────────────────────────────────────────
app = FastAPI(
    title       = "Smart Irrigation API",
    description = "RL-based irrigation controller for wheat farming",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Load model on startup ────────────────────────────────
MODEL_PATH           = "models/dqn_final.zip"
SCENARIO_RESULTS     = "data/scenario_results.json"
ACTION_LABELS        = ["Skip", "Irrigate 50L", "Irrigate 150L", "Irrigate 300L"]
ACTION_VOLUMES       = [0, 50, 150, 300]

# Global state
rl_model         = None
scenario_data    = None
current_env      = None
simulation_cache = {}

@app.on_event("startup")
async def load_resources():
    global rl_model, scenario_data, current_env

    # Load RL model
    if os.path.exists(MODEL_PATH):
        dummy_env  = IrrigationEnv(scenario="normal", seed=42)
        rl_model   = DQN.load(MODEL_PATH, env=dummy_env)
        print(f"RL model loaded from {MODEL_PATH} ✓")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

    # Load scenario results
    if os.path.exists(SCENARIO_RESULTS):
        with open(SCENARIO_RESULTS, "r") as f:
            scenario_data = json.load(f)
        print("Scenario results loaded ✓")
    else:
        print("Warning: scenario_results.json not found")

    # Initialize environment
    current_env = IrrigationEnv(scenario="normal", seed=42)
    current_env.reset()
    print("Environment initialized ✓")


# ── Request/Response Models ──────────────────────────────
class PredictRequest(BaseModel):
    soil_moisture_surface : float = 0.27
    soil_moisture_root    : float = 0.25
    soil_moisture_deep    : float = 0.30
    temperature_c         : float = 20.0
    humidity_pct          : float = 55.0
    rainfall_mm           : float = 0.0
    forecast_24h_mm       : float = 0.0
    forecast_72h_mm       : float = 0.0
    crop_stage            : int   = 1
    days_progress         : float = 0.3
    electricity_price     : float = 0.5

class SimulateRequest(BaseModel):
    scenario  : str = "normal"   # normal | drought | wet
    agent     : str = "rl"       # rl | fixed | threshold
    seed      : int = 42


# ── Endpoints ────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model_loaded": rl_model is not None,
        "scenarios"   : list(scenario_data.keys()) if scenario_data else []
    }


@app.get("/weather")
def weather():
    """Fetch live weather from Open-Meteo for Multan, Pakistan"""
    data = get_live_weather()
    return data


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Given current farm state, return RL agent's recommended action.
    """
    if rl_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build observation vector (must match training order)
    obs = np.array([
        req.soil_moisture_surface,
        req.soil_moisture_root,
        req.soil_moisture_deep,
        np.clip((req.temperature_c + 5) / 50.0, 0, 1),
        np.clip(req.humidity_pct / 100.0, 0, 1),
        np.clip(req.rainfall_mm / 20.0, 0, 1),
        np.clip(req.forecast_24h_mm / 50.0, 0, 1),
        np.clip(req.forecast_72h_mm / 100.0, 0, 1),
        req.crop_stage / 5.0,
        np.clip(req.days_progress, 0, 1),
        req.electricity_price,
    ], dtype=np.float32)

    action, _ = rl_model.predict(obs, deterministic=True)
    action     = int(action)

    # Build explanation
    explanation = _explain_action(action, req)

    return {
        "action"        : action,
        "label"         : ACTION_LABELS[action],
        "volume_litres" : ACTION_VOLUMES[action],
        "explanation"   : explanation,
        "obs_received"  : obs.tolist(),
    }


@app.post("/simulate")
def simulate(req: SimulateRequest):
    """
    Run a full 120-day season simulation.
    Returns daily summaries for plotting.
    """
    if rl_model is None and req.agent == "rl":
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Select agent
    if req.agent == "rl":
        agent      = rl_model
        agent_type = "rl"
    elif req.agent == "fixed":
        agent      = FixedScheduleBaseline()
        agent_type = "baseline"
    else:
        agent      = ThresholdBaseline(threshold=0.28)
        agent_type = "baseline"

    # Run simulation
    env  = IrrigationEnv(scenario=req.scenario, seed=req.seed)
    obs, info = env.reset()
    done = False
    step = 0

    daily_moisture  = []
    daily_actions   = []
    daily_rewards   = []
    total_water     = 0
    total_reward    = 0

    while not done:
        if agent_type == "baseline":
            action = agent.predict(obs, step)
        else:
            action, _ = agent.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done  = terminated or truncated
        step += 1
        total_water  += ACTION_VOLUMES[int(action)]
        total_reward += reward

        # Log once per day (every 12 steps)
        if step % 12 == 0:
            daily_moisture.append(round(info["root_moisture"], 3))
            daily_actions.append(int(action))
            daily_rewards.append(round(reward, 3))

    return {
        "scenario"       : req.scenario,
        "agent"          : req.agent,
        "total_water_L"  : total_water,
        "total_reward"   : round(total_reward, 2),
        "final_yield"    : round(info["yield_estimate"], 1),
        "days"           : list(range(1, len(daily_moisture) + 1)),
        "moisture"       : daily_moisture,
        "actions"        : daily_actions,
        "rewards"        : daily_rewards,
    }


@app.get("/report/scenario")
def scenario_report():
    """Return pre-computed scenario analysis results"""
    if scenario_data is None:
        raise HTTPException(status_code=404, detail="Scenario data not found")
    return scenario_data


@app.get("/dashboard")
def dashboard():
    """
    Return current farm state summary using live weather + RL prediction.
    """
    # Get live weather
    weather  = get_live_weather()

    # Build a representative current state
    req = PredictRequest(
        temperature_c    = weather["temp_c"],
        humidity_pct     = weather["humidity_pct"],
        rainfall_mm      = weather["rainfall_mm"],
        forecast_24h_mm  = weather["forecast_24h_mm"],
        forecast_72h_mm  = weather["forecast_72h_mm"],
    )

    # Get RL recommendation
    prediction = predict(req)

    return {
        "weather"       : weather,
        "recommendation": prediction,
        "farm_state"    : {
            "soil_moisture_surface": req.soil_moisture_surface,
            "soil_moisture_root"   : req.soil_moisture_root,
            "soil_moisture_deep"   : req.soil_moisture_deep,
            "crop_stage"           : req.crop_stage,
        }
    }


# ── Helper functions ─────────────────────────────────────
def _explain_action(action, req):
    """Generate human readable explanation for the agent's decision"""
    reasons = []

    if action == 0:
        if req.forecast_24h_mm > 5:
            reasons.append(f"rain forecast ({req.forecast_24h_mm}mm in 24h)")
        if req.soil_moisture_root > 0.28:
            reasons.append(f"soil moisture adequate ({req.soil_moisture_root:.2f})")
        return "Skip — " + ", ".join(reasons) if reasons else "Skip — soil moisture sufficient"

    volume = ACTION_VOLUMES[action]
    if req.soil_moisture_root < 0.20:
        reasons.append(f"soil critically dry ({req.soil_moisture_root:.2f})")
    if req.crop_stage == 3:
        reasons.append("crop in flowering stage (high water sensitivity)")
    if req.forecast_24h_mm < 2:
        reasons.append("no rain forecast")

    return f"Irrigate {volume}L — " + ", ".join(reasons) if reasons else f"Irrigate {volume}L"


# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)