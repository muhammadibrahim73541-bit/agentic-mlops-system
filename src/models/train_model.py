import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import glob
import os
import warnings

def train():
    files = glob.glob("data/processed/features_*.csv")
    if not files:
        raise ValueError("No feature files found")
    
    df = pd.read_csv(max(files, key=os.path.getctime))
    print(f"Loaded {len(df)} samples from {max(files, key=os.path.getctime)}")
    
    # Handle small datasets
    if len(df) < 5:
        print(f"WARNING: Only {len(df)} samples. Augmenting data...")
        df = pd.concat([df] * 5, ignore_index=True)
        df['energy_demand_mw'] = df['energy_demand_mw'] + np.random.normal(0, 5, len(df))
    
    X = df.drop('energy_demand_mw', axis=1)
    y = df['energy_demand_mw']
    
    # Split
    test_size = 0.2 if len(X) > 5 else 0.0
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Set MLflow to file-based (no SQLite conflicts)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("energy-demand-forecasting")
    
    with mlflow.start_run():
        # Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        try:
            r2 = r2_score(y_test, preds)
        except:
            r2 = 0.0
        
        # Log to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save locally as backup
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/demand_model.pkl")
        
        # Save feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save metrics to file
        with open("models/metrics.txt", "w") as f:
            f.write(f"MAE: {mae}\nR2: {r2}\nSamples: {len(df)}\n")
            f.write("\nFeature Importance:\n")
            f.write(importance.to_string())
        
        print(f"✅ Model trained and logged to MLflow!")
        print(f"MAE: {mae:.2f} MW")
        print(f"R2: {r2:.3f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"\nTop 3 features:\n{importance.head(3)}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train()