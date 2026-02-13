import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.config import config
from src.utils.logger import logger
from src.utils.mlflow_utils import mlflow_utils
from src.data.preprocessing import load_and_preprocess_data

log = logger.get_logger(__name__)

def train():
    """Train the model and log to MLflow"""
    try:
        mlflow_utils.setup_mlflow()
        log.info("Loading and preprocessing data...")
        X_train, X_val, y_train, y_val, preprocessor = load_and_preprocess_data()
        
        preprocessor_path = config.models_dir / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        log.info(f"Preprocessor saved to {preprocessor_path}")
        
        params = config.xgboost_params
        model = xgb.XGBRegressor(**params)
        
        log.info("Training model...")
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=10,
                 verbose=False)
        
        log.info("Evaluating model...")
        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False)
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        log.info("Logging to MLflow...")
        mlflow_utils.log_params(params)
        mlflow_utils.log_metrics(metrics)
        mlflow_utils.log_model(model, "model")
        
        # Log the preprocessor as an artifact
        import mlflow
        mlflow.log_artifact(str(preprocessor_path), "preprocessor")
        
        mlflow_utils.log_plots(model, X_val, y_val)

        
        # Save Model Locally
        model_path = config.models_dir / config.model_name
        joblib.dump(model, model_path)
        log.info(f"Model saved localy to {model_path}")
        
        log.info("Training pipeline completed successfully.")
        
    except Exception as e:
        log.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    train()
