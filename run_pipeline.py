import sys
import os
from src.models.train_model import train
from src.utils.logger import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

log = logger.get_logger("pipeline")

def main():
    try:
        log.info("Starting House Price Prediction Pipeline...")
        log.info("This will preprocess data, train the model, and log to MLflow.")
        
        train()
        
        log.info("Pipeline finished successfully!")
        
    except Exception as e:
        log.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()