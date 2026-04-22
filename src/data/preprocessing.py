import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import logger
from src.utils.config import config

log = logger.get_logger(__name__)

def _build_preprocessor(df: pd.DataFrame):
    """Create preprocessing pipeline based on dataframe schema."""
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    if 'SalePrice' in numeric_features:
        numeric_features = numeric_features.drop('SalePrice')

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


def preprocess_data(df: pd.DataFrame, is_train: bool = True, preprocessor=None):
    try:
        log.info("Starting data preprocessing...")
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)

        if is_train:
            preprocessor = _build_preprocessor(df)
            X = df.drop('SalePrice', axis=1)
            y = df['SalePrice']
            X_processed = preprocessor.fit_transform(X)
            log.info(f"Data shape after preprocessing: {X_processed.shape}")
            return X_processed, y, preprocessor
        else:
            if preprocessor is None:
                raise ValueError("Preprocessor is required when is_train=False.")
            X = df.drop('SalePrice', axis=1)
            y = df['SalePrice']
            X_processed = preprocessor.transform(X)
            log.info(f"Validation data shape after preprocessing: {X_processed.shape}")
            return X_processed, y
            
    except Exception as e:
        log.error(f"Error in preprocessing: {str(e)}")
        raise

def load_and_preprocess_data():
    try:
        train_path = config.raw_data_dir / config.train_file
        
        log.info(f"Loading data from {train_path}")
        train_df = pd.read_csv(train_path)

        train_split_df, val_split_df = train_test_split(
            train_df, test_size=config.test_size, random_state=config.random_state
        )

        X_train, y_train, preprocessor = preprocess_data(train_split_df, is_train=True)
        X_val, y_val = preprocess_data(val_split_df, is_train=False, preprocessor=preprocessor)
        
        return X_train, X_val, y_train, y_val, preprocessor
        
    except Exception as e:
        log.error(f"Error loading data: {str(e)}")
        raise
