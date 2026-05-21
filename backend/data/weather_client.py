import requests
import numpy as np

def get_live_weather(lat=30.18, lon=71.47):
    """
    Fetch real forecast from Open-Meteo for Multan, Pakistan.
    Free — no API key needed.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"     : lat,
        "longitude"    : lon,
        "hourly"       : "temperature_2m,precipitation,relativehumidity_2m",
        "forecast_days": 3,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()["hourly"]

        current = {
            "temp_c"         : round(data["temperature_2m"][0], 1),
            "humidity_pct"   : round(data["relativehumidity_2m"][0], 1),
            "rainfall_mm"    : round(data["precipitation"][0], 2),
            "forecast_24h_mm": round(sum(data["precipitation"][:24]), 2),
            "forecast_72h_mm": round(sum(data["precipitation"][:72]), 2),
            "source"         : "live"
        }
        return current

    except Exception as e:
        print(f"Live forecast failed: {e} — using synthetic fallback")
        return {
            "temp_c"         : 25.0,
            "humidity_pct"   : 50.0,
            "rainfall_mm"    : 0.0,
            "forecast_24h_mm": 0.0,
            "forecast_72h_mm": 0.0,
            "source"         : "fallback"
        }