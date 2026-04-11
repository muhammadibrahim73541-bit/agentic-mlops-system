from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.utils.schema import FeatureSchema, SchemaManager, DANISH_CITIES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = "data/weather_energy.db"
MODEL_PATH = "models/demand_model.pkl"
SCHEMA_PATH = "models/schema_latest.json"

app = FastAPI(
    title="Danish Energy Demand Forecasting API",
    description="Production MLOps Pipeline with W&B tracking",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@contextmanager
def get_db():
    """Context manager for database connections"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize database with proper schema"""
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                city TEXT,
                predicted_demand REAL NOT NULL,
                temp REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                hour INTEGER,
                is_weekend INTEGER,
                is_holiday INTEGER,
                model_version TEXT,
                schema_version TEXT
            )
        ''')
        conn.commit()
        logger.info("Database initialized")

@app.on_event("startup")
async def startup_event():
    init_db()
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}")
    else:
        logger.info(f"Model loaded from {MODEL_PATH}")

class PredictionRequest(BaseModel):
    """Strictly validated prediction request"""
    temp: float = Field(..., ge=-20, le=40, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity %")
    pressure: float = Field(..., ge=950, le=1050, description="Pressure hPa")
    wind_speed: float = Field(..., ge=0, le=50, description="Wind speed m/s")
    hour: int = Field(..., ge=0, le=23, description="Hour of day")
    is_weekend: int = Field(0, ge=0, le=1, description="Is weekend")
    is_holiday: int = Field(0, ge=0, le=1, description="Is Danish holiday")
    month: int = Field(..., ge=1, le=12, description="Month")
    city: Optional[str] = Field(None, description="City name (optional)")
    
    @field_validator('humidity')
    @classmethod
    def validate_humidity(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Humidity must be between 0-100')
        return v

class PredictionResponse(BaseModel):
    """Prediction response with interpretation"""
    predicted_demand_mw: float
    unit: str = "Megawatts"
    interpretation: str
    confidence: str
    weather_score: float
    city: Optional[str]
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    schema_loaded: bool
    supported_cities: List[str]

def get_model():
    """Load model with error handling"""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train model first."
        )
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

def calculate_derived_features(data: PredictionRequest) -> FeatureSchema:
    """Calculate all derived features for model input"""
    return FeatureSchema(
        temp=data.temp,
        humidity=data.humidity,
        pressure=data.pressure,
        wind_speed=data.wind_speed,
        hour=data.hour,
        is_weekend=data.is_weekend,
        is_holiday=data.is_holiday,
        temp_squared=data.temp ** 2,
        temp_humidity=data.temp * data.humidity,
        hour_sin=np.sin(2 * np.pi * data.hour / 24),
        hour_cos=np.cos(2 * np.pi * data.hour / 24),
        month=data.month
    )

def get_interpretation(demand: float) -> tuple:
    """Generate human-readable interpretation"""
    if demand > 1000:
        return ("Critical Peak - Grid Stress", "critical", "high")
    elif demand > 800:
        return ("High Demand - Industrial Peak", "high", "high")
    elif demand > 600:
        return ("Moderate-High Demand", "moderate", "medium")
    elif demand > 400:
        return ("Normal Demand", "normal", "high")
    else:
        return ("Low Demand - Off Peak", "low", "high")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=os.path.exists(MODEL_PATH),
        schema_loaded=os.path.exists(SCHEMA_PATH),
        supported_cities=list(DANISH_CITIES.keys())
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    """Predict energy demand with full schema validation"""
    try:
        model = get_model()
        features = calculate_derived_features(data)
        feature_array = np.array([features.to_model_array()])
        
        # Schema validation
        if os.path.exists(SCHEMA_PATH):
            try:
                SchemaManager.validate_features_against_schema(
                    FeatureSchema.get_feature_names(), 
                    SCHEMA_PATH
                )
            except ValueError as e:
                logger.error(f"Schema validation failed: {e}")
                raise HTTPException(status_code=500, detail=f"Feature mismatch: {e}")
        
        prediction = model.predict(feature_array)[0]
        weather_score = min(100, max(0, abs(data.temp - 18) * 3 + data.wind_speed * 2))
        interp, level, conf = get_interpretation(prediction)
        
        # Store in database
        try:
            with get_db() as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO predictions 
                    (timestamp, city, predicted_demand, temp, humidity, pressure, 
                     wind_speed, hour, is_weekend, is_holiday, model_version, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(), data.city, float(prediction),
                    data.temp, data.humidity, data.pressure, data.wind_speed,
                    data.hour, data.is_weekend, data.is_holiday, "v2.0", "v2.0"
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Database error: {e}")
        
        return PredictionResponse(
            predicted_demand_mw=round(float(prediction), 2),
            unit="Megawatts",
            interpretation=interp,
            confidence=conf,
            weather_score=round(weather_score, 1),
            city=data.city,
            model_version="v2.0",
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predictions")
async def get_predictions(limit: int = 100):
    """Get prediction history"""
    try:
        with get_db() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?",
                conn, params=(limit,)
            )
            return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cities")
async def get_cities():
    """Get supported Danish cities"""
    return {
        "cities": [
            {"name": city, "region": info["region"], "base_temp": info["base_temp"]}
            for city, info in DANISH_CITIES.items()
        ]
    }

@app.get("/model-info")
async def get_model_info():
    """Get current model metadata"""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not found")
    
    model = joblib.load(MODEL_PATH)
    schema_info = SchemaManager.load_schema(SCHEMA_PATH) if os.path.exists(SCHEMA_PATH) else {}
    
    importance = []
    if hasattr(model, 'feature_importances_'):
        importance = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(
                schema_info.get('feature_names', []),
                model.feature_importances_
            )
        ]
        importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return {
        "model_type": type(model).__name__,
        "metrics": schema_info.get('metadata', {}).get('metrics'),
        "feature_importance": importance[:5]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)