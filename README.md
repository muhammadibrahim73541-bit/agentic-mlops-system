# 🌦️ Weather-Crypto MLOps Pipeline

**Live Data → ML Pipeline → FastAPI → Streamlit → GitHub Actions CI/CD**

## Quick Start (GitHub Codespaces)
1. Click green "Code" button → "Codespaces" → "New codespace"
2. In terminal: `pip install -r requirements.txt`
3. Set secret: `export OPENWEATHER_API_KEY=your_key`
4. Run pipeline: `python src/ingestion/fetch_data.py`
5. Train: `python src/models/train_model.py`
6. Start API: `uvicorn api.main:app --reload`
7. Start Frontend: `streamlit run frontend/app.py`

## Architecture
- **Ingestion**: OpenWeatherMap + CoinGecko (free APIs)
- **Features**: Weather severity score vs Crypto volatility
- **Model**: Random Forest (MLflow tracked)
- **API**: FastAPI with auto-docs at `/docs`
- **CI/CD**: GitHub Actions (every 6 hours)
- **Deploy**: Docker + Docker Compose

## Endpoints
- `GET /` - Health check
- `POST /predict` - Get volatility prediction
- `POST /ingest` - Trigger data pipeline