import sqlite3
import pandas as pd
from datetime import datetime
import os

class WeatherCryptoDB:
    def __init__(self, db_path="data/weather_crypto.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_tables()
    
    def init_tables(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS weather_raw
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      city TEXT, timestamp TEXT, temp REAL, 
                      humidity REAL, pressure REAL, weather TEXT,
                      wind_speed REAL, ingestion_time TEXT)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS crypto_raw
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      coin TEXT, timestamp TEXT, price REAL,
                      change_24h REAL, volume REAL, ingestion_time TEXT)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      prediction_time TEXT, temp REAL, humidity REAL,
                      weather_score REAL, predicted_volatility INTEGER,
                      probability REAL, model_version TEXT)''')
        
        conn.commit()
        conn.close()
    
    def insert_weather(self, df):
        conn = sqlite3.connect(self.db_path)
        df['ingestion_time'] = datetime.now().isoformat()
        df.to_sql('weather_raw', conn, if_exists='append', index=False)
        conn.close()
    
    def insert_crypto(self, df):
        conn = sqlite3.connect(self.db_path)
        df['ingestion_time'] = datetime.now().isoformat()
        df.to_sql('crypto_raw', conn, if_exists='append', index=False)
        conn.close()

db = WeatherCryptoDB()