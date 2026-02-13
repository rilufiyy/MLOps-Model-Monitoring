import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    try:
        r = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {r.status_code} - {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return False

def test_inference_mode():
    payload = {
        "features": {
            "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 65.0, "LotArea": 8450, "Street": "Pave", "Alley": "NA", 
            "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl", 
            "Neighborhood": "CollgCr", "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam", "HouseStyle": "2Story", 
            "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2003, "YearRemodAdd": 2003, "RoofStyle": "Gable", "RoofMatl": "CompShg", 
            "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd", "MasVnrType": "BrkFace", "MasVnrArea": 196.0, "ExterQual": "Gd", 
            "ExterCond": "TA", "Foundation": "PConc", "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No", "BsmtFinType1": "GLQ", 
            "BsmtFinSF1": 706, "BsmtFinType2": "Unf", "BsmtFinSF2": 0, "BsmtUnfSF": 150, "TotalBsmtSF": 856, "Heating": "GasA", 
            "HeatingQC": "Ex", "CentralAir": "Y", "Electrical": "SBrkr", "1stFlrSF": 856, "2ndFlrSF": 854, "LowQualFinSF": 0, 
            "GrLivArea": 1710, "BsmtFullBath": 1, "BsmtHalfBath": 0, "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3, 
            "KitchenAbvGr": 1, "KitchenQual": "Gd", "TotRmsAbvGrd": 8, "Functional": "Typ", "Fireplaces": 0, "FireplaceQu": "NA", 
            "GarageType": "Attchd", "GarageYrBlt": 2003.0, "GarageFinish": "RFn", "GarageCars": 2, "GarageArea": 548, 
            "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 61, "EnclosedPorch": 0, 
            "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": "NA", "Fence": "NA", "MiscFeature": "NA", "MiscVal": 0, 
            "MoSold": 2, "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal"
        }
    }
    
    # Test Normal Inference
    try:
        r = requests.post(f"{BASE_URL}/predict?mode=inference", json=payload)
        print(f"Inference Mode: {r.status_code}")
        if r.status_code == 200:
            print("Inference Success:", r.json())
        else:
            print("Inference Failed:", r.text)
        return r.status_code == 200
    except Exception as e:
        print(f"Inference Mode Error: {e}")
        return False

def test_retrain_mode_validation():
    # Payload without SalePrice
    payload = {
        "features": {
             "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 65.0, "LotArea": 8450, "Street": "Pave", "Alley": "NA", 
            "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl", 
            "Neighborhood": "CollgCr", "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam", "HouseStyle": "2Story", 
            "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2003, "YearRemodAdd": 2003, "RoofStyle": "Gable", "RoofMatl": "CompShg", 
            "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd", "MasVnrType": "BrkFace", "MasVnrArea": 196.0, "ExterQual": "Gd", 
            "ExterCond": "TA", "Foundation": "PConc", "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No", "BsmtFinType1": "GLQ", 
            "BsmtFinSF1": 706, "BsmtFinType2": "Unf", "BsmtFinSF2": 0, "BsmtUnfSF": 150, "TotalBsmtSF": 856, "Heating": "GasA", 
            "HeatingQC": "Ex", "CentralAir": "Y", "Electrical": "SBrkr", "1stFlrSF": 856, "2ndFlrSF": 854, "LowQualFinSF": 0, 
            "GrLivArea": 1710, "BsmtFullBath": 1, "BsmtHalfBath": 0, "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3, 
            "KitchenAbvGr": 1, "KitchenQual": "Gd", "TotRmsAbvGrd": 8, "Functional": "Typ", "Fireplaces": 0, "FireplaceQu": "NA", 
            "GarageType": "Attchd", "GarageYrBlt": 2003.0, "GarageFinish": "RFn", "GarageCars": 2, "GarageArea": 548, 
            "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 61, "EnclosedPorch": 0, 
            "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": "NA", "Fence": "NA", "MiscFeature": "NA", "MiscVal": 0, 
            "MoSold": 2, "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal"
        }
    }
    
    # Needs SalePrice
    try:
        r = requests.post(f"{BASE_URL}/predict?mode=retrain", json=payload)
        print(f"Retrain Validation (Missing SalePrice): {r.status_code}")
        if r.status_code == 400:
            print("Validation Success: Got 400 as expected.")
            return True
        else:
            print(f"Validation Failed: Expected 400, got {r.status_code}")
            return False
    except Exception as e:
        print(f"Retrain Validation Error: {e}")
        return False

def test_retrain_mode_success():
    payload = {
        "features": {
             "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 65.0, "LotArea": 8450, "Street": "Pave", "Alley": "NA", 
            "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl", 
            "Neighborhood": "CollgCr", "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam", "HouseStyle": "2Story", 
            "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2003, "YearRemodAdd": 2003, "RoofStyle": "Gable", "RoofMatl": "CompShg", 
            "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd", "MasVnrType": "BrkFace", "MasVnrArea": 196.0, "ExterQual": "Gd", 
            "ExterCond": "TA", "Foundation": "PConc", "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No", "BsmtFinType1": "GLQ", 
            "BsmtFinSF1": 706, "BsmtFinType2": "Unf", "BsmtFinSF2": 0, "BsmtUnfSF": 150, "TotalBsmtSF": 856, "Heating": "GasA", 
            "HeatingQC": "Ex", "CentralAir": "Y", "Electrical": "SBrkr", "1stFlrSF": 856, "2ndFlrSF": 854, "LowQualFinSF": 0, 
            "GrLivArea": 1710, "BsmtFullBath": 1, "BsmtHalfBath": 0, "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3, 
            "KitchenAbvGr": 1, "KitchenQual": "Gd", "TotRmsAbvGrd": 8, "Functional": "Typ", "Fireplaces": 0, "FireplaceQu": "NA", 
            "GarageType": "Attchd", "GarageYrBlt": 2003.0, "GarageFinish": "RFn", "GarageCars": 2, "GarageArea": 548, 
            "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 61, "EnclosedPorch": 0, 
            "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": "NA", "Fence": "NA", "MiscFeature": "NA", "MiscVal": 0, 
            "MoSold": 2, "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal"
        },
        "SalePrice": 200000.0
    }
    
    try:
        r = requests.post(f"{BASE_URL}/predict?mode=retrain", json=payload)
        print(f"Retrain Mode (With SalePrice): {r.status_code}")
        if r.status_code == 200:
            print("Retrain Mode Success:", r.json())
            return True
        else:
            print("Retrain Mode Failed:", r.text)
            return False
    except Exception as e:
        print(f"Retrain Mode Error: {e}")
        return False

if __name__ == "__main__":
    print("Waiting for API to start...")
    # Health check loop
    for _ in range(30):
        if test_health():
            break
        time.sleep(2)
    else:
        print("API did not start in time. Exiting.")
        sys.exit(1)
        
    success = True
    success &= test_inference_mode()
    success &= test_retrain_mode_validation()
    success &= test_retrain_mode_success()
    
    if success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1)
