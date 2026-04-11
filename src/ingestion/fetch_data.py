import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY", "demo")
        # 4 Major Danish Cities
        self.cities = {
            "Copenhagen": {"lat": 55.6761, "lon": 12.5683},
            "Aarhus": {"lat": 56.1629, "lon": 10.2039},
            "Odense": {"lat": 55.4038, "lon": 10.4024},
            "Aalborg": {"lat": 57.0482, "lon": 9.9194}
        }
        
    def fetch_weather(self, city_name, coords):
        """Fetch weather for Danish city - Returns FAHRENHEIT"""
        if self.weather_api_key != "demo":
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": coords["lat"],
                "lon": coords["lon"],
                "appid": self.weather_api_key,
                "units": "imperial"  # ← FAHRENHEIT
            }
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                return {
                    "city": city_name,
                    "timestamp": datetime.now().isoformat(),
                    "temp_f": data["main"]["temp"],  # ← Fahrenheit
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "weather": data["weather"][0]["main"],
                    "wind_speed": data["wind"]["speed"],
                    "country": "DK"
                }
            except Exception as e:
                logger.warning(f"API failed for {city_name}: {e}")
        
        # Mock data with realistic Danish weather patterns (in Fahrenheit)
        np.random.seed()
        mock_temps = {
            "Copenhagen": 45,   # ~7°C
            "Aarhus": 43,       # ~6°C  
            "Odense": 46,       # ~8°C)
            "Aalborg": 41       # ~5°C)
        }
        base = mock_temps.get(city_name, 44)
        
        return {
            "city": city_name,
            "timestamp": datetime.now().isoformat(),
            "temp_f": base + np.random.normal(0, 5),
            "humidity": np.random.randint(60, 85),
            "pressure": 1013 + np.random.normal(0, 3),
            "weather": "Clouds",
            "wind_speed": max(0, np.random.normal(12, 5)),
            "country": "DK"
        }
    
    def calculate_danish_energy_demand(self, weather_df):
        """
        Realistic Danish energy demand model
        Cold weather = High heating demand (Denmark uses district heating)
        """
        demands = []
        for _, row in weather_df.iterrows():
            temp_f = row['temp_f']
            
            # Convert F to rough heating degree days logic
            # Base: 65°F (18°C) - below this, heating needed
            if temp_f < 65:
                heating_factor = (65 - temp_f) * 2.5
            else:
                heating_factor = 0
            
            # Danish-specific: District heating, wind power variability
            base_demand = 450  # MW base load for city
            
            # Population factor (Copenhagen bigger than Aalborg)
            city_pop_factor = {
                "Copenhagen": 1.5,
                "Aarhus": 1.2,
                "Odense": 1.0,
                "Aalborg": 0.9
            }.get(row['city'], 1.0)
            
            # Time of day (business hours)
            hour = datetime.now().hour
            hourly_factor = 1.3 if 8 <= hour <= 18 else 1.0
            
            # Weekend reduction
            weekday = datetime.now().weekday()
            weekend_factor = 0.85 if weekday >= 5 else 1.0
            
            demand = (base_demand + heating_factor * city_pop_factor) * hourly_factor * weekend_factor
            demand += np.random.normal(0, 8)  # Realistic noise
            
            demands.append({
                "timestamp": row['timestamp'],
                "city": row['city'],
                "temp_f": temp_f,
                "temp_c": round((temp_f - 32) * 5/9, 1),  # Store both
                "humidity": row['humidity'],
                "pressure": row['pressure'],
                "wind_speed": row['wind_speed'],
                "energy_demand_mw": max(200, demand),
                "heating_required": 1 if temp_f < 65 else 0,
                "hour": hour,
                "is_weekend": 1 if weekday >= 5 else 0,
                "country": "DK"
            })
        
        return pd.DataFrame(demands)
    
    def run(self):
        os.makedirs("data/raw", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Fetch weather for all Danish cities
        weather_data = []
        for city_name, coords in self.cities.items():
            data = self.fetch_weather(city_name, coords)
            if data:
                weather_data.append(data)
        
        if weather_data:
            df_weather = pd.DataFrame(weather_data)
            df_energy = self.calculate_danish_energy_demand(df_weather)
            
            # Save
            output_path = f"data/raw/denmark_energy_{ts}.csv"
            df_energy.to_csv(output_path, index=False)
            logger.info(f"Saved {output_path}")
            logger.info(f"Average demand: {df_energy['energy_demand_mw'].mean():.1f} MW")
            logger.info(f"Temperature range: {df_energy['temp_f'].min():.1f}°F - {df_energy['temp_f'].max():.1f}°F")
            
        return ts

if __name__ == "__main__":
    DataIngestion().run()