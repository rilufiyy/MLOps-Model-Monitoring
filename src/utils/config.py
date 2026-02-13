import os
from pathlib import Path

class config:
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    models_dir = base_dir / "models"
    logs_dir = base_dir / "logs"
    mlruns_dir = base_dir / "mlruns"
    
    train_file = "train.csv"
    test_file = "test.csv"
    processed_train = "train_processed.csv"
    processed_test = "test_processed.csv"
    monitoring_data = "monitoring_data.csv"
    
    model_name = "house_price_model.pkl"
    random_state = 42
    test_size = 0.2
    
    xgboost_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': random_state,
        'n_jobs': -1
    }
    
    mlflow_tracking_uri = "file:///app/mlruns" if os.getenv("DOCKER_ENV") else mlruns_dir.as_uri()
    mlflow_experiment_name = "house_price_prediction"
    
    # API settings
    api_host = "0.0.0.0"
    api_port = 8000
    
    # Logging
    log_level = "INFO"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file = "mlops_pipeline.log"
    
    @classmethod
    def create_directories(cls):
        directories = [
            cls.data_dir,
            cls.raw_data_dir,
            cls.processed_data_dir,
            cls.models_dir,
            cls.logs_dir,
            cls.mlruns_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)