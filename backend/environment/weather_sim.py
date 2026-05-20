# weather_sim.py
import numpy as np

class WeatherSimulator:
    """
    Synthetic weather generator for Pakistan (Multan region).
    Produces realistic temperature, humidity, and rainfall for training.
    Also supports fetching real forecast from Open-Meteo API (no key needed).
    """

    # Monthly averages for Multan, Punjab, Pakistan
    # [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]
    MONTHLY_TEMP_C   = [12, 15, 22, 29, 34, 37, 35, 34, 31, 26, 19, 13]
    MONTHLY_HUMID    = [55, 48, 40, 35, 32, 38, 58, 65, 55, 42, 48, 55]
    MONTHLY_RAIN_MM  = [14,  9, 12,  8,  8, 10, 48, 42, 12,  2,  2,  8]
   

    def __init__(self, scenario="normal", seed=42):
        """
        scenario: "normal" | "drought" | "wet"
        """
        self.rng      = np.random.default_rng(seed)
        self.scenario = scenario
        self.step_count = 0   # counts 2h timesteps since season start

    def reset(self, scenario=None):
        if scenario:
            self.scenario = scenario
        self.step_count = 0

    def get_weather(self):
        """
        Returns weather for current 2-hour window.
        Returns dict: temp_c, humidity_pct, rainfall_mm,
                      forecast_rain_24h, forecast_rain_72h
        """
        day_of_season = self.step_count        # was self.step_count / 12.0
        # Map season day (0-120) to calendar month starting from Nov (sowing)
        calendar_day  = (day_of_season + 305) % 365  # Nov 1 = day 305
        month_idx     = int(calendar_day / 30.4) % 12

        temp_base = self.MONTHLY_TEMP_C[month_idx]
        hum_base  = self.MONTHLY_HUMID[month_idx]
        rain_base = self.MONTHLY_RAIN_MM[month_idx]

        # Apply scenario multipliers
        if self.scenario == "drought":
            rain_base *= 0.35
            temp_base += 2.5
        elif self.scenario == "wet":
            rain_base *= 1.9
            temp_base -= 1.5

        

        # Diurnal temperature variation (peaks at 2pm = step 7)
        # hour_of_day  = (self.step_count % 12) * 2
        # diurnal      = 6.0 * np.sin(np.pi * (hour_of_day - 6) / 12.0)
        # temp_c       = float(temp_base + diurnal + self.rng.normal(0, 1.5))

        # humidity_pct = float(np.clip(
        #     hum_base + self.rng.normal(0, 5) - diurnal * 1.2,
        #     20, 95
        # ))

        temp_c       = float(temp_base + self.rng.normal(0, 2.0))
        humidity_pct = float(np.clip(
            hum_base + self.rng.normal(0, 5), 20, 95))

        # Rainfall: sporadic — most 2h windows have zero rain
        rain_prob    = (rain_base / 30.0) * 0.08   # probability per 2h window
        rainfall_mm  = 0.0
        if self.rng.random() < rain_prob:
            rainfall_mm = float(self.rng.exponential(rain_base * 0.3))

        # Forecast: look ahead 12 and 36 steps, add noise
        # forecast_24h = self._forecast_rain(12, rain_base, rain_prob)
        # forecast_72h = self._forecast_rain(36, rain_base, rain_prob)
        
        forecast_24h = self._forecast_rain(1, rain_base, rain_prob)  # was 12
        forecast_72h = self._forecast_rain(3, rain_base, rain_prob)  # was 36

        self.step_count += 1

        return {
            "temp_c"         : round(temp_c, 1),
            "humidity_pct"   : round(humidity_pct, 1),
            "rainfall_mm"    : round(rainfall_mm, 2),
            "forecast_24h_mm": round(forecast_24h, 2),
            "forecast_72h_mm": round(forecast_72h, 2),
            "month_idx"      : month_idx,
        }

    def _forecast_rain(self, steps_ahead, rain_base, rain_prob):
        """Imperfect forecast — adds noise to simulate forecast error"""
        expected = 0.0
        for _ in range(steps_ahead):
            if self.rng.random() < rain_prob * 1.1:
                expected += self.rng.exponential(rain_base * 0.3)
        noise = self.rng.normal(1.0, 0.2)
        return max(0.0, expected * noise)

    @staticmethod
    def fetch_live_forecast(lat=30.18, lon=71.47):
        """
        Fetch real 3-day forecast from Open-Meteo (no API key needed).
        Returns summary dict for the current and next 2 days.
        """
        import requests
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude"  : lat,
            "longitude" : lon,
            "hourly"    : "temperature_2m,precipitation,relativehumidity_2m",
            "forecast_days": 3,
        }
        try:
            r = requests.get(url, params=params, timeout=5)
            data = r.json()["hourly"]
            return {
                "temp_c"         : round(np.mean(data["temperature_2m"][:2]), 1),
                "humidity_pct"   : round(np.mean(data["relativehumidity_2m"][:2]), 1),
                "rainfall_mm"    : round(sum(data["precipitation"][:2]), 2),
                "forecast_24h_mm": round(sum(data["precipitation"][:24]), 2),
                "forecast_72h_mm": round(sum(data["precipitation"][:72]), 2),
            }
        except Exception as e:
            print(f"Live forecast failed ({e}), using synthetic.")
            return None

print("WeatherSimulator loaded ✓")