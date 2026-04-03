import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

class FeatureBuilder:
    def __init__(self):
        self.raw_path = "data/raw"
        self.processed_path = "data/processed"
        os.makedirs(self.processed_path, exist_ok=True)
    
    def load_latest(self):
        files = glob.glob(f"{self.raw_path}/energy_demand_*.csv")
        if not files:
            raise ValueError("No energy data found. Run ingestion first.")
        return pd.read_csv(max(files, key=os.path.getctime))
    
    def engineer(self):
        df = self.load_latest()
        
        # Feature engineering
        df['temp_squared'] = df['temp'] ** 2  # Non-linear temperature effect
        df['temp_humidity'] = df['temp'] * df['humidity']  # Interaction
        df['demand_per_temp'] = df['energy_demand_mw'] / (df['temp'] + 10)  # Efficiency metric
        
        # Lag features (previous hour - simulate if we had time series)
        df['demand_lag1'] = df['energy_demand_mw'].shift(1).fillna(df['energy_demand_mw'].mean())
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        features = df[[
            'temp', 'humidity', 'pressure', 'wind_speed', 'hour',
            'is_weekend', 'temp_squared', 'temp_humidity', 
            'hour_sin', 'hour_cos', 'energy_demand_mw'
        ]]
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"{self.processed_path}/features_{ts}.csv"
        features.to_csv(output, index=False)
        print(f"Features saved: {output}")
        return output, features

if __name__ == "__main__":
    import numpy as np  # Add this import
    FeatureBuilder().engineer()