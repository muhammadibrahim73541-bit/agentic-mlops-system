python3 << 'PYEOF'
content = '''import pandas as pd
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
    
    # Handle small datasets
    if len(df) < 5:
        df = pd.concat([df] * 5, ignore_index=True)
        df['energy_demand_mw'] = df['energy_demand_mw'] + np.random.normal(0, 5, len(df))
    
    X = df.drop('energy_demand_mw', axis=1)
    y = df['energy_demand_mw']
    
    # Train/test split
    test_size = 0.2 if len(X) > 5 else 0.0
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Initialize W&B run
    run = wandb.init(
        project="danish-energy-forecast",
        name=f"model-training-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}",
        job_type="training",
        reinit=True
    )
    
    # Model configuration
    config = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_samples": len(df),
        "n_features": X.shape[1]
    }
    
    if tune:
        # Simple hyperparameter tuning
        config["n_estimators"] = 200
        config["max_depth"] = 15
    
    wandb.config.update(config)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"]
    )
    model.fit(X_train, y_train)
    
    # Metrics
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    try:
        r2 = r2_score(y_test, preds)
    except:
        r2 = 0.0
    
    # Log metrics to W&B
    wandb.log({
        "MAE": mae,
        "R2": r2,
        "feature_importance": {col: imp for col, imp in zip(X.columns, model.feature_importances_)}
    })
    
    # Save model locally
    os.makedirs("models", exist_ok=True)
    model_path = "models/demand_model.pkl"
    joblib.dump(model, model_path)
    
    # Log model artifact to W&B
    artifact = wandb.Artifact(
        name="energy-demand-model",
        type="model",
        description=f"RandomForest model - MAE: {mae:.2f}, R2: {r2:.3f}"
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    
    # Save metrics to JSON for pipeline artifact upload
    metrics = {
        "MAE": float(mae),
        "R2": float(r2),
        "n_samples": len(df),
        "n_features": X.shape[1],
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    with open("models/metrics.txt", "w") as f:
        f.write(f"MAE: {mae}\\nR2: {r2}\\nSamples: {len(df)}\\n")
        f.write(importance.to_string())
    
    print(f"✅ Model trained!")
    print(f"MAE: {mae:.2f} MW | R2: {r2:.3f}")
    
    wandb.finish()
    return model, metrics

if __name__ == "__main__":
    import sys
    tune = "--tune" in sys.argv
    train(tune=tune)
'''

with open('src/models/train_model.py', 'w') as f:
    f.write(content)
print("W&B train_model.py written successfully")
PYEOF