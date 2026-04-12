import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import wandb
import glob
import os
import json

def train(tune=False):
    files = glob.glob("data/processed/features_*.csv")
    if not files:
        raise ValueError("No feature files found")
    
    df = pd.read_csv(max(files, key=os.path.getctime))
    
    if len(df) < 5:
        df = pd.concat([df] * 5, ignore_index=True)
        df['energy_demand_mw'] = df['energy_demand_mw'] + np.random.normal(0, 5, len(df))
    
    X = df.drop('energy_demand_mw', axis=1)
    y = df['energy_demand_mw']
    
    test_size = 0.2 if len(X) > 5 else 0.0
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    wandb.init(project="danish-energy-forecast", job_type="training")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds) if len(y_test) > 1 else 0.0
    
    wandb.log({"MAE": mae, "R2": r2})
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/demand_model.pkl")
    
    with open("models/metrics.json", "w") as f:
        json.dump({"MAE": mae, "R2": r2}, f)
    
    wandb.finish()
    print(f"✅ Model trained! MAE: {mae:.2f} MW | R2: {r2:.3f}")

if __name__ == "__main__":
    import sys
    train("--tune" in sys.argv)
