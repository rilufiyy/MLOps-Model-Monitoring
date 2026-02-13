import streamlit as st
import requests
import json
import pandas as pd

# API Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

st.title("House Price Prediction System")
st.markdown("""
This interface allows you to predict house prices using the trained model.
You can choose between **Standard Inference** and **Retrain Mode**.
""")

# Sidebar for Mode Selection
mode = st.sidebar.radio(
    "Select Mode",
    options=["Inference Biasa", "Inference & Retrain"],
    help="Select 'Inference & Retrain' to provide the actual Sale Price for monitoring and retraining purposes."
)

api_mode = "inference" if mode == "Inference Biasa" else "retrain"

st.header(f"Mode: {mode}")

# Input Form
with st.form("prediction_form"):
    st.subheader("House Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MSSubClass = st.number_input("MSSubClass", value=60)
        MSZoning = st.selectbox("MSZoning", ["RL", "RM", "C (all)", "FV", "RH"])
        LotFrontage = st.number_input("LotFrontage", value=65.0)
        LotArea = st.number_input("LotArea", value=8450)
        Street = st.selectbox("Street", ["Pave", "Grvl"])
        Alley = st.selectbox("Alley", ["NA", "Grvl", "Pave"])
        LotShape = st.selectbox("LotShape", ["Reg", "IR1", "IR2", "IR3"])
        LandContour = st.selectbox("LandContour", ["Lvl", "Bnk", "HLS", "Low"])
        Utilities = st.selectbox("Utilities", ["AllPub", "NoSeWa"])
        LotConfig = st.selectbox("LotConfig", ["Inside", "Corner", "CulDSac", "FR2", "FR3"])
        LandSlope = st.selectbox("LandSlope", ["Gtl", "Mod", "Sev"])
        Neighborhood = st.selectbox("Neighborhood", ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"])

    with col2:
        Condition1 = st.selectbox("Condition1", ["Norm", "Feedr", "PosN", "Artery", "RRAe", "RRNn", "RRAn", "PosA", "RRNe"])
        Condition2 = st.selectbox("Condition2", ["Norm", "Artery", "RDN", "Feedr", "RRNn", "PosN", "PosA", "RRAn", "RRAe"])
        BldgType = st.selectbox("BldgType", ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"])
        HouseStyle = st.selectbox("HouseStyle", ["2Story", "1Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"])
        OverallQual = st.slider("OverallQual", 1, 10, 7)
        OverallCond = st.slider("OverallCond", 1, 10, 5)
        YearBuilt = st.number_input("YearBuilt", value=2003)
        YearRemodAdd = st.number_input("YearRemodAdd", value=2003)
        RoofStyle = st.selectbox("RoofStyle", ["Gable", "Hip", "Gambrel", "Mansard", "Flat", "Shed"])
        RoofMatl = st.selectbox("RoofMatl", ["CompShg", "WdShngl", "Metal", "WdShake", "Membran", "Tar&Grv", "Roll", "ClyTile"])
        Exterior1st = st.selectbox("Exterior1st", ["VinylSd", "MetalSd", "Wd Sdng", "HdBoard", "BrkFace", "WdShing", "CemntBd", "Plywood", "AsbShng", "Stucco", "BrkComm", "AsphShn", "Stone", "ImStucc", "CBlock"])
        Exterior2nd = st.selectbox("Exterior2nd", ["VinylSd", "MetalSd", "Wd Sdng", "HdBoard", "BrkFace", "Wd Shng", "CmentBd", "Plywood", "AsbShng", "Stucco", "Brk Cmn", "AsphShn", "Stone", "ImStucc", "CBlock", "Other"])

    with col3:
        MasVnrType = st.selectbox("MasVnrType", ["BrkFace", "None", "Stone", "BrkCmn"])
        MasVnrArea = st.number_input("MasVnrArea", value=196.0)
        ExterQual = st.selectbox("ExterQual", ["Gd", "TA", "Ex", "Fa"])
        ExterCond = st.selectbox("ExterCond", ["TA", "Gd", "Fa", "Po", "Ex"])
        Foundation = st.selectbox("Foundation", ["PConc", "CBlock", "BrkTil", "Wood", "Slab", "Stone"])
        BsmtQual = st.selectbox("BsmtQual", ["Gd", "TA", "Ex", "Fa", "NA"])
        BsmtCond = st.selectbox("BsmtCond", ["TA", "Gd", "Fa", "Po", "NA"])
        BsmtExposure = st.selectbox("BsmtExposure", ["No", "Gd", "Mn", "Av", "NA"])
        BsmtFinType1 = st.selectbox("BsmtFinType1", ["GLQ", "ALQ", "Unf", "Rec", "BLQ", "LwQ", "NA"])
        BsmtFinSF1 = st.number_input("BsmtFinSF1", value=706)
        BsmtFinType2 = st.selectbox("BsmtFinType2", ["Unf", "GLQ", "ALQ", "Rec", "BLQ", "LwQ", "NA"])
        BsmtFinSF2 = st.number_input("BsmtFinSF2", value=0)
        BsmtUnfSF = st.number_input("BsmtUnfSF", value=150)
        TotalBsmtSF = st.number_input("TotalBsmtSF", value=856)

    st.markdown("---")
    st.subheader("Other Features (Defaults applied for brevity)")
    # Using defaults for the rest to keep the UI clean-ish, but sending them in payload
    # In a real app, all these would be inputs.
    
    # Target Variable Input (Conditional)
    sale_price_input = None
    if api_mode == "retrain":
        st.markdown("### 🎯 Target Variable (Required for Retraining)")
        sale_price_input = st.number_input("Actual Sale Price ($)", value=200000.0)

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Construct Payload
    payload = {
        "features": {
            "MSSubClass": MSSubClass,
            "MSZoning": MSZoning,
            "LotFrontage": LotFrontage,
            "LotArea": LotArea,
            "Street": Street,
            "Alley": Alley,
            "LotShape": LotShape,
            "LandContour": LandContour,
            "Utilities": Utilities,
            "LotConfig": LotConfig,
            "LandSlope": LandSlope,
            "Neighborhood": Neighborhood,
            "Condition1": Condition1,
            "Condition2": Condition2,
            "BldgType": BldgType,
            "HouseStyle": HouseStyle,
            "OverallQual": OverallQual,
            "OverallCond": OverallCond,
            "YearBuilt": YearBuilt,
            "YearRemodAdd": YearRemodAdd,
            "RoofStyle": RoofStyle,
            "RoofMatl": RoofMatl,
            "Exterior1st": Exterior1st,
            "Exterior2nd": Exterior2nd,
            "MasVnrType": MasVnrType,
            "MasVnrArea": MasVnrArea,
            "ExterQual": ExterQual,
            "ExterCond": ExterCond,
            "Foundation": Foundation,
            "BsmtQual": BsmtQual,
            "BsmtCond": BsmtCond,
            "BsmtExposure": BsmtExposure,
            "BsmtFinType1": BsmtFinType1,
            "BsmtFinSF1": BsmtFinSF1,
            "BsmtFinType2": BsmtFinType2,
            "BsmtFinSF2": BsmtFinSF2,
            "BsmtUnfSF": BsmtUnfSF,
            "TotalBsmtSF": TotalBsmtSF,
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
            "SaleCondition": "Normal"
        }
    }
    
    if api_mode == "retrain":
        payload["SalePrice"] = sale_price_input

    try:
        with st.spinner("Requesting prediction..."):
            response = requests.post(f"{API_URL}/predict?mode={api_mode}", json=payload)
            
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Price: ${result['prediction']:,.2f}")
            if api_mode == "retrain":
                st.info("Data has been submitted for monitoring.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Is it running?")
    except Exception as e:
        st.error(f"An error occurred: {e}")