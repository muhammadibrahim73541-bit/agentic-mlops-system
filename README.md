# ⚡ Energy AI - Danish Grid Forecasting

**MLOps Pipeline for Energy Demand Prediction**  
*MSc BDS Exam Project - Copenhagen Business School*

[![MLOps Pipeline](https://github.com/YOUR_USERNAME/agentic-mlops-system/actions/workflows/pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/agentic-mlops-system/actions/workflows/pipeline.yml)

![W&B](https://img.shields.io/badge/Weights_&_Biases-FF6F00?style=flat&logo=weightsandbiases&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5B2E?style=flat)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat&logo=next.js&logoColor=white)

## 🎯 Project Overview

Production-grade MLOps system forecasting energy demand for **4 Danish cities** (Copenhagen, Aarhus, Odense, Aalborg) using weather data and XGBoost models.

**Key Features:**
- ✅ **Schema Validation** - Prevents training-serving skew
- ✅ **Weights & Biases** - Full experiment tracking & artifact management
- ✅ **Danish Energy Physics** - Realistic heating/cooling models
- ✅ **Next.js Dashboard** - Professional dark-mode interface
- ✅ **Dockerized** - Multi-service deployment
- ✅ **CI/CD** - Automated training every 6 hours

## 🏗️ Architecture
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   OpenWeather   │──────>  Data Ingestion  │──────>  Feature Eng.  │
│   Energidata.dk │      │  (4 DK Cities)   │      │  (Schema Enf.)  │
└─────────────────┘      └──────────────────┘      └─────────────────┘
│
┌──────────────────┐                │
│   Weights &      │<───────────────┘
│   Biases (W&B)   │    (XGBoost + CV)
└──────────────────┘
│
┌──────────▼──────────┐
│   FastAPI Backend   │
│   (Validation +     │
│    Schema Check)    │
└──────────┬──────────┘
│
┌──────────▼──────────┐
│   Next.js Frontend  │
│   (Dark Dashboard)  │
└─────────────────────┘




```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/agentic-mlops-system.git
cd agentic-mlops-system

# 2. Set environment variables
cp .env.example .env
# Edit .env:
# - WANDB_API_KEY=your_key_here
# - OPENWEATHER_API_KEY=your_key_here (optional, has demo fallback)

# 3. Run with Docker Compose (Recommended)
docker-compose up --build

# OR: Manual Setup
cd backend && pip install -r requirements.txt
python -m src.ingestion.fetch_data
python -m src.features.build_features  
python -m src.models.train_model
uvicorn api.main:app --reload

# Frontend (new terminal)
cd frontend && npm install && npm run dev