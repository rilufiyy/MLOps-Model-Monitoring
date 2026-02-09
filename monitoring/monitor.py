import numpy as np
import pandas as pd
from scipy import stats
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from run_pipeline import main as train_pipeline
    from src.utils.logger import logger
    from src.utils.config import config
except ImportError as e:
    print(f"Error: Could not import run_pipeline. Make sure you are in the project root or monitoring directory.\nDetail: {e}")
    sys.exit(1)

log = logger.get_logger("monitoring")

def generate_data(n_samples=1000, drift=False):
    """
    Generate dummy feature data.
    """
    if drift:
        # Simulate drift: Mean shift from 0 to 2, Std dev change from 1 to 1.5
        data = np.random.normal(loc=2.0, scale=1.5, size=n_samples)
    else:
        # Reference distribution: Standard Normal
        data = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    
    return data

def detect_drift(reference_data, current_data, threshold=0.05):
    """
    Perform Kolmogorov-Smirnov test to detect data drift.
    """
    statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    log.info(f"drift check: p-value={p_value:.5f}, threshold={threshold}")
    
    if p_value < threshold:
        return True, p_value
    return False, p_value

def load_reference_data():
    """Load reference data (training data) for drift detection."""
    # In a real scenario, this should load the actual training data used for the model
    # For now, we simulate it or load a sample if available
    train_path = config.data_dir / config.train_file
    if train_path.exists():
         df = pd.read_csv(train_path)
         # Assuming 'SalePrice' is the target, we might monitor features or the target itself.
         # For simplicity, let's return SalePrice as the reference
         return df['SalePrice'].values
    return generate_data(n_samples=1000, drift=False)

def check_and_retrain(new_data_point):
    """
    Save new data point, check for drift, and trigger retraining if needed.
    """
    monitoring_path = config.data_dir / config.monitoring_data
    
    # Create DataFrame from the new data point
    new_df = pd.DataFrame([new_data_point])
    
    # Save to CSV (append mode)
    header = not monitoring_path.exists()
    new_df.to_csv(monitoring_path, mode='a', header=header, index=False)
    
    log.info(f"New data point saved to {monitoring_path}")
    
    # Load gathered monitoring data
    monitoring_df = pd.read_csv(monitoring_path)
    
    # Perform check only if we have enough data (e.g., > 10 samples) to catch drift
    # In production, this might be a larger batch size
    if len(monitoring_df) < 10:
        log.info(f"Not enough data to check for drift yet ({len(monitoring_df)} samples).")
        return
        
    current_data = monitoring_df['SalePrice'].values
    reference_data = load_reference_data()
    
    # Check for Drift
    is_drifted, p_val = detect_drift(reference_data, current_data)
    
    if is_drifted:
        log.warning(f"DATA DRIFT DETECTED! (p-value: {p_val:.5f}). Triggering retraining...")
        # Trigger Retraining
        train_pipeline()
        
        # Optional: Clear monitoring data or archive it after retraining
        # os.remove(monitoring_path) 
    else:
        log.info(f"No drift detected (p-value: {p_val:.5f}). System is healthy.")

def monitor_system(simulate_drift=False):
    log.info("Starting system monitoring...")
    
    # Load Reference Data 
    log.info("Loading reference data...")
    reference_data = load_reference_data()
    
    # Collect New Data (Simulated incoming stream)
    log.info(f"Collecting new data (simulate_drift={simulate_drift})...")
    new_data = generate_data(n_samples=500, drift=simulate_drift)
    
    # Check for Drift
    is_drifted, p_val = detect_drift(reference_data, new_data)
    
    if is_drifted:
        log.warning(f"DATA DRIFT DETECTED! (p-value: {p_val:.5f}). Triggering retraining...")
        # Trigger Retraining
        train_pipeline()
    else:
        log.info("No drift detected. System is healthy.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Data Drift")
    parser.add_argument("--drift", action="store_true", help="Simulate data drift to trigger retraining")
    args = parser.parse_args()
    
    monitor_system(simulate_drift=args.drift)
