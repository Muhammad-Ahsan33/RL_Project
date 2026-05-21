# Smart Irrigation System using Reinforcement Learning
### CT-469 Reinforcement Learning — Course Project (P09)
**Spring 2026 | Final Year AI Specialisation**

---

## Project Overview

Agriculture accounts for 70% of global freshwater consumption. Traditional fixed-schedule irrigation wastes 30–50% of water through over-irrigation and inability to adapt to changing weather conditions.

This project implements a **Deep Q-Network (DQN) agent** that learns optimal irrigation scheduling for wheat farming in Multan, Pakistan. The agent observes soil moisture sensors, weather forecasts, and crop growth stage to make daily irrigation decisions — maximizing crop health while minimizing water consumption and energy costs.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RL Algorithm | DQN (Stable-Baselines3) |
| Environment | Custom Gymnasium environment |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Weather API | Open-Meteo (free, no key needed) |
| Language | Python 3.10+ |

---

## Project Structure
RL_Project/
├── backend/
│   ├── environment/
│   │   ├── irrigation_env.py    # Custom Gymnasium environment
│   │   ├── soil_model.py        # 3-layer soil water dynamics
│   │   ├── crop_model.py        # Wheat growth and stress model
│   │   └── weather_sim.py       # Synthetic weather generator
│   ├── baselines/
│   │   └── baseline_agents.py   # Fixed schedule + threshold agents
│   ├── data/
│   │   ├── weather_client.py    # Open-Meteo API client
│   │   └── scenario_results.json
│   ├── models/
│   │   └── dqn_final.zip        # Pre-trained DQN model
│   ├── logs/
│   │   └── training_log.csv
│   └── main.py                  # FastAPI application
└── frontend/
└── app.py                   # Streamlit dashboard

---

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** | Soil moisture (3 depths), temperature, humidity, rainfall, 24h/72h forecast, crop stage, season progress, electricity price |
| **Actions** | 0=Skip, 1=100L, 2=300L, 3=600L per day |
| **Reward** | Moisture reward − overwater penalty − water cost + drought bonus − stress penalty |
| **Episode** | 120 steps (1 step = 1 day, full wheat season) |

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOURUSERNAME/smart-irrigation-rl.git
cd smart-irrigation-rl
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Install frontend dependencies
```bash
cd ../frontend
pip install streamlit requests plotly pandas
```

### 5. Download pre-trained model
Download `dqn_final.zip` from [Google Drive link] and place it in `backend/models/`

---

## How to Run Training

Open Google Colab and upload the notebook `Agent_Training.ipynb`


To reproduce exact results use `seed=42` in all cells.

---

## How to Run the Web App

### Terminal 1 — Start backend
```bash
cd backend
venv\Scripts\activate
python main.py
```
Backend runs at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### Terminal 2 — Start frontend
```bash
cd frontend
streamlit run app.py
```
Dashboard opens at `http://localhost:8501`

---

## How to Reproduce Results

Run this in Colab after training:

```python
# Fixed seed guarantees identical results
env = IrrigationEnv(scenario="normal", seed=42)
model = DQN.load("models/dqn_final.zip", env=env)



## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API status check |
| `/weather` | GET | Live weather for Multan |
| `/predict` | POST | Get RL agent recommendation |
| `/simulate` | POST | Run full season simulation |
| `/report/scenario` | GET | Pre-computed scenario results |
| `/dashboard` | GET | Current farm state + recommendation |


## Known Limitations

- Soil ET model may underestimate winter water requirements
- Water usage unrealistically low due to conservative winter ET
- Single crop (wheat) only different crops require retuning
- No real soil sensor hardware integration

---
