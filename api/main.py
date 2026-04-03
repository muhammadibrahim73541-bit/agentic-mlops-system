from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sqlite3
import os
import logging
from datetime import datetime
from typing import Optional, List
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Energy Demand Forecasting API",
    description="MLOps Pipeline for weather-based energy demand prediction",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_PATH = "data/weather_crypto.db"
MODEL_PATH = "models/demand_model.pkl"

def get_db():
    """Get database connection"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables"""
    conn = get_db()
    c = conn.cursor()
    
    # Predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        predicted_demand REAL,
        actual_demand REAL,
        temp REAL,
        humidity REAL,
        pressure REAL,
        wind_speed REAL,
        hour INTEGER,
        is_weekend INTEGER,
        model_version TEXT DEFAULT 'v1.0'
    )''')
    
    # Raw data table
    c.execute('''CREATE TABLE IF NOT EXISTS raw_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ingestion_time TEXT,
        city TEXT,
        temp REAL,
        humidity REAL,
        pressure REAL,
        wind_speed REAL,
        energy_demand REAL
    )''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")

# Request/Response models
class PredictionRequest(BaseModel):
    temp: float
    humidity: float
    pressure: float
    wind_speed: float
    hour: int
    is_weekend: int = 0
    
    class Config:
        schema_extra = {
            "example": {
                "temp": 20.0,
                "humidity": 50.0,
                "pressure": 1013.0,
                "wind_speed": 5.0,
                "hour": 12,
                "is_weekend": 0
            }
        }

class PredictionResponse(BaseModel):
    predicted_demand_mw: float
    unit: str
    interpretation: str
    temp: float
    hour: int
    weather_score: float
    confidence: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str

# Helper functions
def calculate_weather_score(temp: float, humidity: float, wind_speed: float) -> float:
    """Calculate weather severity score"""
    temp_extreme = abs(temp - 20)
    score = (temp_extreme * 2 + wind_speed * 3 + (100 - humidity)) / 10
    return score

def get_model():
    """Load model with error handling"""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train model first: python src/models/train_model.py"
        )
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

# API Endpoints

@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML"""
    if os.path.exists("docs/dashboard.html"):
        return FileResponse("docs/dashboard.html")
    elif os.path.exists("dashboard.html"):
        return FileResponse("dashboard.html")
    else:
        return {"message": "API is running. Dashboard not found."}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=os.path.exists(MODEL_PATH),
        version="2.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    """
    Predict energy demand based on weather parameters.
    Stores prediction in database for monitoring.
    """
    try:
        model = get_model()
        
        # Calculate derived features
        weather_score = calculate_weather_score(data.temp, data.humidity, data.wind_speed)
        temp_sq = data.temp ** 2
        temp_hum = data.temp * data.humidity
        hour_sin = np.sin(2 * np.pi * data.hour / 24)
        hour_cos = np.cos(2 * np.pi * data.hour / 24)
        
        # Feature vector (must match training order)
        features = [[
            data.temp, data.humidity, data.pressure, data.wind_speed,
            data.hour, data.is_weekend, temp_sq, temp_hum,
            hour_sin, hour_cos
        ]]
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Determine interpretation
        if prediction > 600:
            interpretation = "High demand - Peak load"
            confidence = "high"
        elif prediction > 450:
            interpretation = "Moderate demand"
            confidence = "medium"
        else:
            interpretation = "Normal demand"
            confidence = "high"
        
        # Store in database
        try:
            conn = get_db()
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions 
                (timestamp, predicted_demand, temp, humidity, pressure, wind_speed, hour, is_weekend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                prediction,
                data.temp,
                data.humidity,
                data.pressure,
                data.wind_speed,
                data.hour,
                data.is_weekend
            ))
            conn.commit()
            conn.close()
            logger.info(f"Prediction stored: {prediction:.2f} MW")
        except Exception as e:
            logger.error(f"Database error: {e}")
            # Continue even if DB fails
        
        return PredictionResponse(
            predicted_demand_mw=round(prediction, 2),
            unit="Megawatts",
            interpretation=interpretation,
            temp=data.temp,
            hour=data.hour,
            weather_score=round(weather_score, 2),
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predictions")
async def get_predictions(limit: int = 100):
    """Get prediction history from database"""
    try:
        conn = get_db()
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", 
            conn, 
            params=(limit,)
        )
        conn.close()
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch predictions")

@app.get("/latest-data")
async def get_latest_data():
    """Get latest raw data for charts"""
    try:
        import glob
        files = glob.glob("data/raw/energy_demand_*.csv")
        if not files:
            return {"error": "No data files found"}
        
        latest = max(files, key=os.path.getctime)
        df = pd.read_csv(latest)
        
        return {
            "cities": df.to_dict('records'),
            "timestamp": datetime.now().isoformat(),
            "count": len(df)
        }
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    try:
        conn = get_db()
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        
        if len(df) == 0:
            return {"message": "No predictions yet"}
        
        metrics = {
            "total_predictions": len(df),
            "avg_demand": float(df['predicted_demand'].mean()),
            "max_demand": float(df['predicted_demand'].max()),
            "min_demand": float(df['predicted_demand'].min())
        }
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """Trigger data pipeline (runs in background)"""
    def run_pipeline():
        try:
            import sys
            sys.path.append('src')
            from ingestion.fetch_data import DataIngestion
            from features.build_features import FeatureBuilder
            
            logger.info("Starting data ingestion...")
            DataIngestion().run()
            logger.info("Building features...")
            FeatureBuilder().engineer()
            logger.info("Pipeline complete")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
    
    background_tasks.add_task(run_pipeline)
    return {"message": "Pipeline triggered in background"}

@app.get("/model-info")
async def get_model_info():
    """Get model metadata"""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not found")
    
    try:
        model = joblib.load(MODEL_PATH)
        features = ['temp', 'humidity', 'pressure', 'wind_speed', 'hour', 
                   'is_weekend', 'temp_sq', 'temp_hum', 'hour_sin', 'hour_cos']
        
        importance = model.feature_importances_.tolist()
        
        return {
            "model_type": type(model).__name__,
            "n_features": len(features),
            "feature_importance": dict(zip(features, importance)),
            "model_path": MODEL_PATH,
            "last_modified": datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)