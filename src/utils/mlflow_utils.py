import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from src.utils.config import config
from src.utils.logger import logger

log = logger.get_logger(__name__)

class mlflow_utils:
    @staticmethod
    def setup_mlflow():
        """Setup MLflow tracking URI and experiment"""
        try:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment_name)
            log.info(f"MLflow tracking URI set to: {config.mlflow_tracking_uri}")
            log.info(f"MLflow experiment set to: {config.mlflow_experiment_name}")
        except Exception as e:
            log.error(f"Error setting up MLflow: {str(e)}")
            raise
    
    @staticmethod
    def log_params(params: dict):
        """Log parameters to MLflow"""
        try:
            mlflow.log_params(params)
            log.info(f"Logged parameters: {params}")
        except Exception as e:
            log.error(f"Error logging parameters: {str(e)}")
            raise
    
    @staticmethod
    def log_metrics(metrics: dict):
        """Log metrics to MLflow"""
        try:
            mlflow.log_metrics(metrics)
            log.info(f"Logged metrics: {metrics}")
        except Exception as e:
            log.error(f"Error logging metrics: {str(e)}")
            raise
    
    @staticmethod
    def log_model(model, artifact_path: str):
        """Log model to MLflow"""
        try:
            mlflow.sklearn.log_model(model, artifact_path)
            log.info(f"Logged model to artifact path: {artifact_path}")
        except Exception as e:
            log.error(f"Error logging model: {str(e)}")
            raise

    @staticmethod
    def log_plots(model, X_val, y_val, model_type='xgboost'):
        """
        Generate and log plots to MLflow.
        """
        try:
            log.info("Generating and logging plots...")
            
            # Create a temporary directory for plots
            plot_dir = "temp_plots"
            os.makedirs(plot_dir, exist_ok=True)
            
            # Actual vs Predicted Scatter Plot
            predictions = model.predict(X_val)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=y_val, y=predictions)
            plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")
            plt.title("Actual vs Predicted House Prices")
            scatter_path = os.path.join(plot_dir, "actual_vs_predicted.png")
            plt.savefig(scatter_path)
            plt.close()
            mlflow.log_artifact(scatter_path, "plots")
            
            # Residuals Plot
            residuals = y_val - predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True)
            plt.title("Residuals Distribution")
            plt.xlabel("Residual (Actual - Predicted)")
            residuals_path = os.path.join(plot_dir, "residuals_distribution.png")
            plt.savefig(residuals_path)
            plt.close()
            mlflow.log_artifact(residuals_path, "plots")
            
            # Feature Importance (Specific for XGBoost/Tree models)
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:20] 
                
                plt.title("Top 20 Feature Importances")
                plt.bar(range(len(indices)), importances[indices], align='center')
                plt.xticks(range(len(indices)), indices, rotation=90)
                plt.tight_layout()
                
                importance_path = os.path.join(plot_dir, "feature_importance.png")
                plt.savefig(importance_path)
                plt.close()
                mlflow.log_artifact(importance_path, "plots")

            log.info("Plots logged successfully.")
            
            
        except Exception as e:
            log.error(f"Error logging plots: {str(e)}")