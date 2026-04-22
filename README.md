# MLOps-Model-Monitoring
This project implements an end-to-end MLOps pipeline for house price prediction using MLflow, FastAPI, and Docker. It focuses on modular code design, experiment tracking, model monitoring, structured logging, and production-ready model deployment.

## Project Objectives
- Build modular and maintainable ML codebase
- Implement MLflow experiment tracking & model versioning
- Develop structured logging & monitoring system
- Deploy model serving API using FastAPI
- Containerize application using Docker & Docker Compose

## Project Structure
<img width="542" height="772" alt="image" src="https://github.com/user-attachments/assets/0b8bad13-ef72-4990-926e-12f864cb4819" />

## MLOps Pipeline Overview
### 1. Modular Code Architecture
- Separation of concerns:
  - data/ → preprocessing & feature engineering
  - models/ → training & evaluation
  - utils/ → configuration, logging, MLflow
  - app/ → API serving
- Clean and maintainable production-style structure

### 2. MLflow Experiment Tracking
MLflow digunakan untuk:
- Logging model parameters
- Tracking training metrics:
  - RMSE
  - MAE
  - R² Score
- Saving model artifacts
- Experiment versioning
- MLflow artifacts tersimpan di: mlruns/
- Untuk menjalankan MLflow UI: mlflow ui
- Lalu akses di: http://localhost:5000

### 3. Structured Logging System
- Logging dikonfigurasi melalui: src/utils/logger.py
- Logging mencakup: Data preprocessing events, Model training events, Evaluation results, Prediction requests, Error handling
- Log file disimpan di: logs/mlops_pipeline.log

### 4. FastAPI Model Serving
FastAPI digunakan untuk production model serving.
- Lokasi utama: src/app/
- Contoh endpoint:
  - GET / → health check
  - POST /predict → predict house price
- Request divalidasi menggunakan Pydantic schema untuk memastikan input consistency.
- Menjalankan API secara lokal: uvicorn src.app.main:app --reload
- Akses: http://localhost:8000/docs

### 5. Monitoring System
- Folder: monitoring/
- Fungsi monitoring:
  - Logging prediction distribution
  - Saving prediction results
  - Monitoring input drift (basic monitoring)
  - Tracking inference behavior
- Monitoring data disimpan di: data/monitoring_data.csv

### 6. Docker Deployment
- Project sudah dikontainerisasi menggunakan:
  - Dockerfile
  - docker-compose.yml
- Build Docker image: docker build -t house-price-mlops .
- Run with Docker Compose: docker-compose up --build
- Services yang berjalan:
  - FastAPI service
  - Monitoring support
  - MLflow tracking (optional jika dikonfigurasi)
 
## Model Development
- Dataset: House Prices (Kaggle)
- Problem Type: Regression
- Target: SalePrice
- Evaluation Metrics:
  - RMSE
  - MAE
  - R² Score
- Pipeline meliputi:
  1. Data cleaning
  2. Feature engineering
  3. Preprocessing (encoding, scaling)
  4. Model training
  5. Evaluation
  6. Experiment logging
- Model artifacts tersimpan di: models/

## How to Run Locally
### 1️. Clone Repository
git clone https://github.com/your-username/MLOps-Model-Monitoring.git
cd MLOps-Model-Monitoring

### 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

### 3.Install Dependencies
pip install -r requirements.txt

### 4. Train Model
python -m src.models.train_model

### 5.Run API
uvicorn src.app.main:app --reload

## Key MLOps Features Implemented
- Modular production-ready code
- MLflow experiment tracking
- Structured logging system
- Model artifact versioning
- FastAPI serving
- Monitoring system
- Docker containerization

## Future Improvements
- Automated CI/CD pipeline
- Model drift detection (statistical)
- Model registry integration
- Cloud deployment (AWS/GCP/Azure)
- Prometheus + Grafana monitoring

## Author
Sri Lutfiya Dwiyeni

MLOps & Machine Learning Enthusiast
