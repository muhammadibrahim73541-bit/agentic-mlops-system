# ⚡ Energy Demand Forecasting - MLOps Pipeline

**Live Weather Data → ML Pipeline → FastAPI → HTML Dashboard → GitHub Actions CI/CD**

## Live Demo
- **Dashboard**: https://muhammadibrahim73541-bit.github.io/agentic-mlops-system/
- **API Docs**: Run locally (see below)

## Architecture
![Architecture](docs/architecture.png)

- **Ingestion**: OpenWeatherMap API (free tier)
- **Features**: Temperature, humidity, pressure → Energy demand features
- **Model**: Random Forest Regressor (MLflow tracked)
- **API**: FastAPI with auto-docs at `/docs`
- **Frontend**: HTML/CSS/JS dashboard (dark theme)
- **CI/CD**: GitHub Actions (every 6 hours)
- **Deploy**: Docker + GitHub Pages

## Quick Start

### Local Development
```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key (get from openweathermap.org)
export OPENWEATHER_API_KEY=your_key

# 3. Run pipeline
python src/ingestion/fetch_data.py
python src/features/build_features.py
python src/models/train_model.py

# 4. Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. Open dashboard (in new terminal)
# Option A: Open docs/index.html directly in browser
# Option B: python -m http.server 8501 --directory docs
