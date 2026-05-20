# irrigation_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment.soil_model import SoilModel
from environment.crop_model import CropModel
from environment.weather_sim import WeatherSimulator

# Make sure the above cells have been run first
# (SoilModel, CropModel, WeatherSimulator must be in scope)

class IrrigationEnv(gym.Env):
    """
    Custom Gymnasium environment: Smart Irrigation for Wheat (Multan, Pakistan).

    Observation space : Box(11,) — all features normalized to [0, 1]
    Action space      : Discrete(4)
        0 → skip (no irrigation)
        1 → irrigate 50L  (~5mm)
        2 → irrigate 150L (~15mm)
        3 → irrigate 300L (~30mm)

    Episode length    : 1440 steps (120 days × 12 steps/day @ 2h each)
    """

    metadata = {"render_modes": ["human"]}

    # Litres to mm conversion (assuming 10m × 10m field = 100 m²)
    # FIELD_AREA_M2     = 100.0
    FIELD_AREA_M2 = 50.0
    # ACTION_VOLUMES_L  = [0, 50, 150, 300]   # litres per action
    ACTION_VOLUMES_L = [0, 100, 300, 600]
    STEPS_PER_DAY     = 1
    SEASON_DAYS       = 120
    MAX_STEPS         = SEASON_DAYS * STEPS_PER_DAY  # 1440

    # Reward weights — tune these to change agent behaviour
    CONSERVATION_WEIGHT = 0.1    # reward per litre saved vs max
    ENERGY_COST_PER_L   = 0.002  # electricity cost weight per litre pumped
    STRESS_PENALTY_SCALE = 5.0   # multiplier on crop stress penalty

    def __init__(self, scenario="normal", seed=42):
        super().__init__()

        self.scenario    = scenario
        self.base_seed   = seed

        # Sub-components
        self.soil    = SoilModel(seed=seed)
        self.crop    = CropModel()
        self.weather = WeatherSimulator(scenario=scenario, seed=seed)

        # Gymnasium spaces
        self.action_space = spaces.Discrete(4)

        # 11 features (all normalized 0-1):
        # [moisture_surface, moisture_root, moisture_deep,
        #  temp_norm, humidity_norm, rain_now_norm,
        #  forecast_24h_norm, forecast_72h_norm,
        #  crop_stage_norm, days_progress_norm, elec_price_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )

        # Tracking
        self.current_step      = 0
        self.total_water_used  = 0.0
        self.episode_reward    = 0.0
        self.current_weather   = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset all components
        self.soil.reset()
        self.crop.reset()
        self.weather.reset()

        self.current_step     = 0
        self.total_water_used = 0.0
        self.episode_reward   = 0.0

        self.current_weather = self.weather.get_weather()
        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        # 1. Convert action to irrigation amount
        irrigation_l  = self.ACTION_VOLUMES_L[action]
        irrigation_mm = self._litres_to_mm(irrigation_l)

        # 2. Get current weather
        w = self.current_weather

        # 3. Update soil
        new_moisture, runoff = self.soil.step(
            irrigation_mm  = irrigation_mm,
            rainfall_mm    = w["rainfall_mm"],
            temperature_c  = w["temp_c"],
            humidity_pct   = w["humidity_pct"],
        )

        # 4. Update crop
        _, stress_penalty = self.crop.step(
            root_moisture  = float(self.soil.moisture[1]),
            wilting_point  = SoilModel.WILTING_POINT,
            field_capacity = SoilModel.FIELD_CAPACITY,
        )

        # 5. Calculate reward
        reward = self._calculate_reward(
            irrigation_l   = irrigation_l,
            stress_penalty = stress_penalty,
            w              = w,
        )

        # 6. Advance weather for next step
        self.current_step    += 1
        self.total_water_used += irrigation_l
        self.episode_reward  += reward
        self.current_weather  = self.weather.get_weather()

        # 7. Check termination
        terminated = self.crop.is_done or self.current_step >= self.MAX_STEPS
        truncated  = False

        obs  = self._get_obs()
        info = self._get_info()
        info["irrigation_l"]  = irrigation_l
        info["stress_penalty"] = stress_penalty

        return obs, reward, terminated, truncated, info

    # def _calculate_reward(self, irrigation_l, stress_penalty, w):
    #   """
    #   Redesigned reward — penalizes dry soil directly each step
    #   so agent cannot exploit conservation bonus by doing nothing.
    #   """
    #   root_moisture  = float(self.soil.moisture[1])
    #   field_capacity = SoilModel.FIELD_CAPACITY   # 0.35
    #   wilting_point  = SoilModel.WILTING_POINT    # 0.12
    #   optimal        = 0.26

    #   # 1. Moisture maintenance reward — highest when soil is near optimal
    #   if root_moisture >= optimal:
    #       moisture_reward = 1.0
    #   elif root_moisture >= wilting_point:
    #       moisture_reward = (root_moisture - wilting_point) / (optimal - wilting_point)
    #   else:
    #       moisture_reward = -2.0   # severe penalty below wilting point

    #   # 2. Overwatering penalty
    #   overwater_penalty = 0.0
    #   if root_moisture > field_capacity:
    #       overwater_penalty = (root_moisture - field_capacity) * 5.0

    #   # 3. Water cost — small penalty per litre used
    #   water_cost = (irrigation_l / 300.0) * 0.1

    #   # 4. Smart skip bonus — only reward skipping if soil is already healthy
    #   skip_bonus = 0.0
    #   if irrigation_l == 0 and root_moisture >= optimal:
    #       skip_bonus = 0.2
    #   if irrigation_l == 0 and w["forecast_24h_mm"] > 5.0:
    #       skip_bonus += 0.1

    #   # 5. Stress penalty from crop model
    #   stress_term = stress_penalty * 4.0

    #   reward = moisture_reward - overwater_penalty - water_cost + skip_bonus - stress_term
    #   return float(reward)


    def _calculate_reward(self, irrigation_l, stress_penalty, w):
      root_moisture  = float(self.soil.moisture[1])
      wilting_point  = SoilModel.WILTING_POINT    # 0.12
      field_capacity = SoilModel.FIELD_CAPACITY   # 0.35
      optimal        = 0.26

      # 1. Moisture reward — core signal
      if root_moisture >= optimal and root_moisture <= field_capacity:
          moisture_reward = 2.0    # perfect range
      elif root_moisture >= wilting_point:
          # Linear scale between wilting and optimal
          moisture_reward = 2.0 * (root_moisture - wilting_point) / (optimal - wilting_point)
      else:
          moisture_reward = -5.0   # below wilting — severe

      # 2. Overwatering penalty
      overwater_penalty = 0.0
      if root_moisture > field_capacity:
          overwater_penalty = (root_moisture - field_capacity) * 10.0

      # 3. Tiny water cost — barely penalize irrigation
      water_cost = (irrigation_l / 50.0) * 0.05

      # 4. NO skip bonus — remove entirely

      reward = moisture_reward - overwater_penalty - water_cost
      return float(reward)

    

    def _get_obs(self):
        """Build and normalize the 11-feature state vector"""
        m = self.soil.moisture
        w = self.current_weather if self.current_weather else {
            "temp_c": 25, "humidity_pct": 50, "rainfall_mm": 0,
            "forecast_24h_mm": 0, "forecast_72h_mm": 0
        }

        # Electricity price: simulated peak hours (cheaper at night)
        hour = (self.current_step % 12) * 2
        elec_price = 0.3 + 0.7 * np.sin(np.pi * hour / 12.0) ** 2

        obs = np.array([
            # Soil moisture (already 0-1 as fractions)
            float(m[0]),
            float(m[1]),
            float(m[2]),
            # Weather (normalized)
            np.clip((w["temp_c"] + 5) / 50.0, 0, 1),        # -5°C to 45°C
            np.clip(w["humidity_pct"] / 100.0, 0, 1),
            np.clip(w["rainfall_mm"] / 20.0, 0, 1),          # cap at 20mm/2h
            np.clip(w["forecast_24h_mm"] / 50.0, 0, 1),
            np.clip(w["forecast_72h_mm"] / 100.0, 0, 1),
            # Crop state
            self.crop.stage / 5.0,                            # 0 to 1
            np.clip(self.current_step / self.MAX_STEPS, 0, 1),
            # Electricity
            float(elec_price),
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        return {
            "step"            : self.current_step,
            "day"             : self.current_step / self.STEPS_PER_DAY,
            "crop_stage"      : self.crop.stage,
            "crop_stage_name" : self.crop.STAGES[self.crop.stage][0],
            "root_moisture"   : float(self.soil.moisture[1]),
            "total_water_L"   : self.total_water_used,
            "yield_estimate"  : self.crop.get_yield_score(),
            "episode_reward"  : self.episode_reward,
        }

    # def _litres_to_mm(self, litres):
    #     """Convert litres to mm depth over the field"""
    #     return (litres / self.FIELD_AREA_M2) * 1000.0 / 10.0

    def _litres_to_mm(self, litres):
      """Convert litres to mm depth over the field"""
      return (litres / self.FIELD_AREA_M2)  # 1 litre per m² = 1mm

    def render(self, mode="human"):
        info = self._get_info()
        print(
            f"Day {info['day']:5.1f} | Stage: {info['crop_stage_name']:12s} | "
            f"Root moisture: {info['root_moisture']:.3f} | "
            f"Water used: {info['total_water_L']:6.0f}L | "
            f"Yield est: {info['yield_estimate']:5.1f}"
        )

print("IrrigationEnv loaded ✓")