"""
Schema Management for Danish Energy Demand Forecasting
Prevents training-serving skew via strict validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import json
import os


class FeatureSchema(BaseModel):
    """
    Canonical feature schema - ensures training/inference consistency
    THIS FIXES THE FEATURE ORDER BUG
    """
    # Raw inputs
    temp: float = Field(..., ge=-20, le=40, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity %")
    pressure: float = Field(..., ge=950, le=1050, description="Atmospheric pressure hPa")
    wind_speed: float = Field(..., ge=0, le=50, description="Wind speed m/s")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    is_weekend: int = Field(0, ge=0, le=1, description="Weekend flag (0/1)")
    is_holiday: int = Field(0, ge=0, le=1, description="Danish holiday flag (0/1)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    
    # Derived features (calculated from raw inputs)
    temp_squared: float = Field(..., description="Temperature squared (non-linear effect)")
    temp_humidity: float = Field(..., description="Temp-humidity interaction")
    hour_sin: float = Field(..., ge=-1, le=1, description="Cyclical hour encoding (sin)")
    hour_cos: float = Field(..., ge=-1, le=1, description="Cyclical hour encoding (cos)")
    
    @field_validator('humidity')
    @classmethod
    def validate_humidity(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Humidity must be between 0-100%')
        return v
    
    @field_validator('temp')
    @classmethod
    def validate_temp(cls, v):
        if not -20 <= v <= 40:
            raise ValueError('Temperature seems unrealistic for Denmark (-20 to 40°C)')
        return v
    
    @field_validator('pressure')
    @classmethod
    def validate_pressure(cls, v):
        if not 950 <= v <= 1050:
            raise ValueError('Pressure seems unrealistic (950-1050 hPa)')
        return v

    def to_model_array(self) -> List[float]:
        """
        Returns features in EXACT order used during training
        THIS IS THE CRITICAL FIX - prevents feature order mismatch
        """
        return [
            self.temp,           # 0
            self.humidity,       # 1
            self.pressure,       # 2
            self.wind_speed,     # 3
            self.hour,           # 4
            self.is_weekend,     # 5
            self.is_holiday,     # 6
            self.temp_squared,   # 7
            self.temp_humidity,  # 8
            self.hour_sin,       # 9
            self.hour_cos,       # 10
            self.month           # 11
        ]
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Canonical feature order - must match training exactly"""
        return [
            'temp', 'humidity', 'pressure', 'wind_speed', 'hour',
            'is_weekend', 'is_holiday', 'temp_squared', 'temp_humidity',
            'hour_sin', 'hour_cos', 'month'
        ]
    
    @classmethod
    def from_raw_inputs(cls, temp: float, humidity: float, pressure: float,
                       wind_speed: float, hour: int, is_weekend: int = 0,
                       is_holiday: int = 0, month: Optional[int] = None) -> "FeatureSchema":
        """Factory method to create schema from raw inputs (calculates derived features)"""
        if month is None:
            from datetime import datetime
            month = datetime.now().month
            
        import numpy as np
        
        return cls(
            temp=temp,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind_speed,
            hour=hour,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            month=month,
            temp_squared=temp ** 2,
            temp_humidity=temp * humidity,
            hour_sin=np.sin(2 * np.pi * hour / 24),
            hour_cos=np.cos(2 * np.pi * hour / 24)
        )


class SchemaManager:
    """Manages schema persistence to prevent training-serving skew"""
    
    @staticmethod
    def save_schema(path: str, feature_names: List[str], metadata: Dict = None):
        """Save schema alongside model artifact"""
        schema = {
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'metadata': metadata or {},
            'version': '2.0',
            'schema_type': 'danish_energy_forecast'
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(schema, f, indent=2)
    
    @staticmethod
    def load_schema(path: str) -> Dict[str, Any]:
        """Load and validate schema"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Schema file not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def validate_features_against_schema(input_features: List[str], schema_path: str):
        """Validate that input features match training schema exactly"""
        schema = SchemaManager.load_schema(schema_path)
        expected = schema['feature_names']
        
        if input_features != expected:
            raise ValueError(
                f"FEATURE MISMATCH! Training-serving skew detected!\n"
                f"Expected: {expected}\n"
                f"Got:      {input_features}\n"
                f"Missing:  {set(expected) - set(input_features)}\n"
                f"Extra:    {set(input_features) - set(expected)}"
            )


# ============================================================================
# DANISH CITIES CONFIGURATION
# ============================================================================

DANISH_CITIES = {
    'Copenhagen': {
        'lat': 55.6761, 
        'lon': 12.5683, 
        'base_temp': 9.5, 
        'region': 'Capital',
        'population_factor': 1.5  # Largest city
    },
    'Aarhus': {
        'lat': 56.1629, 
        'lon': 10.2039, 
        'base_temp': 8.5, 
        'region': 'Central',
        'population_factor': 1.0
    },
    'Odense': {
        'lat': 55.4038, 
        'lon': 10.4024, 
        'base_temp': 9.0, 
        'region': 'Southern',
        'population_factor': 0.7
    },
    'Aalborg': {
        'lat': 57.0488, 
        'lon': 9.9217, 
        'base_temp': 8.0, 
        'region': 'Northern',
        'population_factor': 0.8
    }
}

# Danish public holidays 2024 (affects industrial demand)
DANISH_HOLIDAYS_2024 = [
    '2024-01-01',  # New Year
    '2024-03-28',  # Maundy Thursday
    '2024-03-29',  # Good Friday
    '2024-04-01',  # Easter Monday
    '2024-04-26',  # General Prayer Day
    '2024-05-09',  # Ascension Day
    '2024-05-10',  # Bank Holiday
    '2024-05-20',  # Whit Monday
    '2024-06-05',  # Constitution Day
    '2024-12-24',  # Christmas Eve
    '2024-12-25',  # Christmas Day
    '2024-12-26',  # Second Day of Christmas
]


def get_city_info(city_name: str) -> Dict[str, Any]:
    """Get configuration for a specific Danish city"""
    if city_name not in DANISH_CITIES:
        raise ValueError(f"Unknown city: {city_name}. Must be one of {list(DANISH_CITIES.keys())}")
    return DANISH_CITIES[city_name]


def is_danish_holiday(date_str: str) -> int:
    """Check if date string (YYYY-MM-DD) is a Danish holiday"""
    return 1 if date_str in DANISH_HOLIDAYS_2024 else 0


# Feature descriptions for documentation
FEATURE_DESCRIPTIONS = {
    'temp': 'Temperature in Celsius',
    'humidity': 'Relative humidity %',
    'pressure': 'Atmospheric pressure in hPa',
    'wind_speed': 'Wind speed in m/s',
    'hour': 'Hour of day (0-23)',
    'is_weekend': 'Binary flag for weekend',
    'is_holiday': 'Binary flag for Danish public holiday',
    'month': 'Month of year (1-12)',
    'temp_squared': 'Non-linear temperature effect (heating/cooling curve)',
    'temp_humidity': 'Interaction: humidity increases cooling load at high temps',
    'hour_sin': 'Cyclical encoding: sin(2π*hour/24)',
    'hour_cos': 'Cyclical encoding: cos(2π*hour/24)'
}
