from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from src.utils.config import config
from src.utils.logger import logger
import os
import sys
from typing import Optional
from fastapi import Query

# Add project root to path to allow importing monitoring
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from monitoring.monitor import check_and_retrain

# Initialize Logger
log = logger.get_logger("api")

app = FastAPI(title="House Price Prediction API", version="1.0.0")

# Global variables for model and preprocessor
model = None
preprocessor = None

class HouseFeatures(BaseModel):
    features: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "MSSubClass": 60,
                        "MSZoning": "RL",
                        "LotFrontage": 65.0,
                        "LotArea": 8450,
                        "Street": "Pave",
                        "Alley": "NA",
                        "LotShape": "Reg",
                        "LandContour": "Lvl",
                        "Utilities": "AllPub",
                        "LotConfig": "Inside",
                        "LandSlope": "Gtl",
                        "Neighborhood": "CollgCr",
                        "Condition1": "Norm",
                        "Condition2": "Norm",
                        "BldgType": "1Fam",
                        "HouseStyle": "2Story",
                        "OverallQual": 7,
                        "OverallCond": 5,
                        "YearBuilt": 2003,
                        "YearRemodAdd": 2003,
                        "RoofStyle": "Gable",
                        "RoofMatl": "CompShg",
                        "Exterior1st": "VinylSd",
                        "Exterior2nd": "VinylSd",
                        "MasVnrType": "BrkFace",
                        "MasVnrArea": 196.0,
                        "ExterQual": "Gd",
                        "ExterCond": "TA",
                        "Foundation": "PConc",
                        "BsmtQual": "Gd",
                        "BsmtCond": "TA",
                        "BsmtExposure": "No",
                        "BsmtFinType1": "GLQ",
                        "BsmtFinSF1": 706,
                        "BsmtFinType2": "Unf",
                        "BsmtFinSF2": 0,
                        "BsmtUnfSF": 150,
                        "TotalBsmtSF": 856,
                        "Heating": "GasA",
                        "HeatingQC": "Ex",
                        "CentralAir": "Y",
                        "Electrical": "SBrkr",
                        "1stFlrSF": 856,
                        "2ndFlrSF": 854,
                        "LowQualFinSF": 0,
                        "GrLivArea": 1710,
                        "BsmtFullBath": 1,
                        "BsmtHalfBath": 0,
                        "FullBath": 2,
                        "HalfBath": 1,
                        "BedroomAbvGr": 3,
                        "KitchenAbvGr": 1,
                        "KitchenQual": "Gd",
                        "TotRmsAbvGrd": 8,
                        "Functional": "Typ",
                        "Fireplaces": 0,
                        "FireplaceQu": "NA",
                        "GarageType": "Attchd",
                        "GarageYrBlt": 2003.0,
                        "GarageFinish": "RFn",
                        "GarageCars": 2,
                        "GarageArea": 548,
                        "GarageQual": "TA",
                        "GarageCond": "TA",
                        "PavedDrive": "Y",
                        "WoodDeckSF": 0,
                        "OpenPorchSF": 61,
                        "EnclosedPorch": 0,
                        "3SsnPorch": 0,
                        "ScreenPorch": 0,
                        "PoolArea": 0,
                        "PoolQC": "NA",
                        "Fence": "NA",
                        "MiscFeature": "NA",
                        "MiscVal": 0,
                        "MoSold": 2,
                        "YrSold": 2008,
                        "SaleType": "WD",
                        "SaleCondition": "Normal",
                        "SalePrice": 208500
                    }
                }
            ]
        }
    }
    
    # Optional field for retraining mode
    SalePrice: Optional[float] = None

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    try:
        model_path = config.models_dir / config.model_name
        preprocessor_path = config.models_dir / "preprocessor.joblib"
        
        if not model_path.exists() or not preprocessor_path.exists():
            log.warning("Model or preprocessor not found. Please train the model first.")
            return

        log.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        log.info(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        
        log.info("Artifacts loaded successfully.")
    except Exception as e:
        log.error(f"Error loading artifacts: {str(e)}")
        raise

@app.post("/predict")
def predict(data: HouseFeatures, mode: str = Query("inference", enum=["inference", "retrain"])):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    
    # Handle Retrain Mode
    if mode == "retrain":
        if data.SalePrice is None:
           raise HTTPException(status_code=400, detail="SalePrice is required for 'retrain' mode.")
        
        # trigger monitoring
        # We need to construct the full data point including the target
        monitoring_data = data.features.copy()
        monitoring_data['SalePrice'] = data.SalePrice
        
        # Fire and forget (or await) - for now synchronous
        try:
            check_and_retrain(monitoring_data)
            log.info("Monitoring check completed.")
        except Exception as mon_e:
            log.error(f"Monitoring failed: {mon_e}")
            # We don't fail the prediction if monitoring fails, just log it

    try:
        input_df = pd.DataFrame([data.features])
        
        processed_data = preprocessor.transform(input_df)
        prediction = model.predict(processed_data)
        
        return {
            "prediction": float(prediction[0]),
            "currency": "USD" 
        }

    except Exception as e:
        log.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
