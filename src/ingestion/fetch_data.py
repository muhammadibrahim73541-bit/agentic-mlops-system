import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY", "demo")
        self.cities = ["London", "New York", "Tokyo", "Singapore"]
        
    def fetch_weather(self, city):
        """Fetch weather data"""
        if self.weather_api_key != "demo":
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {"q": city, "appid": self.weather_api_key, "units": "metric"}
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                return {
                    "city": city,
                    "timestamp": datetime.now().isoformat(),
                    "temp": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "weather": data["weather"][0]["main"],
                    "wind_speed": data["wind"]["speed"]
                }
            except Exception as e:
                logger.warning(f"API failed for {city}, using mock data: {e}")
        
        # Mock data with realistic patterns
        np.random.seed()
        base_temps = {"London": 12, "New York": 18, "Tokyo": 22, "Singapore": 30}
        base = base_temps.get(city, 20)
        
        return {
            "city": city,
            "timestamp": datetime.now().isoformat(),
            "temp": base + np.random.normal(0, 3),
            "humidity": np.random.randint(40, 90),
            "pressure": 1013 + np.random.normal(0, 5),
            "weather": "Clear",
            "wind_speed": max(0, np.random.normal(10, 5))
        }
    
    def generate_energy_demand(self, weather_df):
        """
        Generate synthetic but realistic energy demand based on weather
        Cold (<10C) and Hot (>25C) = High demand
        """
        demands = []
        for _, row in weather_df.iterrows():
            temp = row['temp']
            
            # Base demand
            base_demand = 500  # MW
            
            # Temperature effect (U-curve: high demand at extremes)
            if temp < 10:  # Heating
                temp_factor = (10 - temp) * 15
            elif temp > 25:  # Cooling
                temp_factor = (temp - 25) * 20
            else:  # Comfortable range
                temp_factor = 0
            
            # Humidity factor (humid = more AC)
            humidity_factor = (row['humidity'] - 50) * 0.5 if temp > 20 else 0
            
            # Random business/hourly variation
            hour = datetime.now().hour
            hourly_factor = 50 if 9 <= hour <= 17 else 20  # Business hours
            
            # Weekend vs weekday
            weekday = datetime.now().weekday()
            weekend_factor = 0.8 if weekday >= 5 else 1.0
            
            demand = (base_demand + temp_factor + humidity_factor + hourly_factor) * weekend_factor
            demand += np.random.normal(0, 10)  # Noise
            
            demands.append({
                "timestamp": row['timestamp'],
                "city": row['city'],
                "temp": temp,
                "humidity": row['humidity'],
                "pressure": row['pressure'],
                "wind_speed": row['wind_speed'],
                "energy_demand_mw": max(200, demand),  # Min 200MW
                "hour": hour,
                "is_weekend": 1 if weekday >= 5 else 0
            })
        
        return pd.DataFrame(demands)
    
    def run(self):
        os.makedirs("data/raw", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Fetch weather
        weather_data = [self.fetch_weather(c) for c in self.cities]
        weather_data = [w for w in weather_data if w]
        
        if weather_data:
            df_weather = pd.DataFrame(weather_data)
            
            # Generate energy demand
            df_energy = self.generate_energy_demand(df_weather)
            
            # Save
            output_path = f"data/raw/energy_demand_{ts}.csv"
            df_energy.to_csv(output_path, index=False)
            logger.info(f"Saved {output_path}")
            logger.info(f"Average demand: {df_energy['energy_demand_mw'].mean():.1f} MW")
            
        return ts

if __name__ == "__main__":
    DataIngestion().run()