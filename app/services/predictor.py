import pandas as pd
import numpy as np
from app.core.config import MODEL_PATH
from tensorflow.keras.utils import to_categorical
from fastapi.encoders import jsonable_encoder
from io import StringIO
from tensorflow import keras 
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.aur_roc import roc
from app.schemas.evaluation import evaluate_model
from app.services.shap_explainer import get_shap_importances
from fastapi import UploadFile
from app.schemas.prediction import PredictionRequest
import logging
import ast 
from app.schemas.safe_encoder import safe_jsonable_encoder
logger = logging.getLogger(__name__)

model = keras.models.load_model(MODEL_PATH)

feature_names = [
        'Unit1', 'Gender', 'HospAdmTime', 'Age', 'DBP', 'Temp', 'Glucose', 'Potassium', 'Hct', 'FiO2',
        'Hgb', 'pH', 'BUN', 'WBC', 'Magnesium', 'Creatinine', 'Platelets', 'Calcium', 'PaCO2',
        'BaseExcess', 'Chloride', 'HCO3', 'Phosphate', 'EtCO2', 'SaO2', 'PTT', 'Lactate', 'AST',
        'Alkalinephos', 'Bilirubin_total', 'TroponinI', 'Fibrinogen', 'Bilirubin_direct'
]

# Safe way to parse strings to Python objects

def preprocess_csv(df: pd.DataFrame):
    """
    Preprocess a DataFrame loaded from CSV that was originally a .pkl file with embedded arrays.
    
    Assumes 'X_cont' is a stringified list (e.g., "[0.5, 1.2, 0.3]") and 'label' is an integer class.
    Converts 'X_cont' into a numerical NumPy array and handles categorical features.
    """
    try:
        # --- Process label column ---
        y_test = np.asarray(list(df['label']))
        y_test = to_categorical(y_test)

        # --- Parse 'X_cont' strings back to float arrays ---
        def parse_x_cont(x):
            try:
                # Clean any extra spaces or non-numeric characters
                x = x.strip()
                # Handle array format with possible extra spaces and parse
                if x.startswith("[[") and x.endswith("]]"):
                    return np.array(ast.literal_eval(x), dtype=np.float32)
                else:
                    raise ValueError(f"Invalid array format: {x}")
            except Exception as e:
                logger.error(f"Error parsing 'X_cont' value: {x}. Error: {str(e)}")
                return None  # Return None to mark this row as invalid
        
        # Apply parsing function with error handling
        df['X_cont'] = df['X_cont'].apply(parse_x_cont)
        
        # Remove rows with invalid 'X_cont' (those that couldn't be parsed)
        df = df.dropna(subset=['X_cont'])
        
        # Ensure that the 'X_cont' column contains valid arrays
        if df['X_cont'].isnull().any():
            raise ValueError("Some 'X_cont' values are invalid after parsing.")

        # Stack the valid arrays into a NumPy array
        X_test_cont = np.stack(df['X_cont'].values)  # shape: (samples, sequence_len, num_channels)
        
        # --- Handle categorical features ---
        X_test_cat = df.drop(['X_cont', 'label'], axis=1)
        X_test_cat[X_test_cat.isna()] = np.pi  # Replace NaN with pi for mask layer
        X_test_cat = np.asarray(list(X_test_cat.values), dtype=np.float32)

        return X_test_cont, X_test_cat, y_test

    except Exception as e:
        logger.error(f"Error preprocessing CSV test data: {str(e)}")
        raise ValueError(f"Error preprocessing CSV test data: {str(e)}")




async def predict_from_csv(df: pd.DataFrame):
    try:
        # Process the CSV file
        X_test_cont, X_test_cat, y_test = preprocess_csv(df)
        
        # Ensure consistent float32 dtype
        X_test_cont = X_test_cont.astype('float32')
        X_test_cat = X_test_cat.astype('float32')
        
        # Make predictions
        prediction = model.predict([X_test_cont, X_test_cat])
        pred_probs = prediction[:, 1].tolist()
        
        # Calculate metrics
        results_df, thresh_final, AUC = roc(prediction[:, 1], y_test[:, 1], 'val')
        metrics = evaluate_model(prediction[:, 1], y_test[:, 1], thresh_final, 'val')

        # Define your feature names
        categorical_feature_names = [
            'Unit1', 'Gender', 'HospAdmTime', 'Age', 'DBP', 'Temp', 'Glucose', 
            'Potassium', 'Hct', 'FiO2', 'Hgb', 'pH', 'BUN', 'WBC', 'Magnesium', 
            'Creatinine', 'Platelets', 'Calcium', 'PaCO2', 'BaseExcess', 'Chloride', 
            'HCO3', 'Phosphate', 'EtCO2', 'SaO2', 'PTT', 'Lactate', 'AST',
            'Alkalinephos', 'Bilirubin_total', 'TroponinI', 'Fibrinogen', 'Bilirubin_direct'
        ]
        
        # Time-series features (adjust these to match your actual time-series features)
        time_series_feature_names = ['X_cont_1', 'X_cont_2', 'X_cont_3', 'X_cont_4', 'X_cont_5']

        # SHAP explanation with safety checks
        shap_results = {}
        if len(X_test_cont) > 10:
            try:
                shap_results = get_shap_importances(
                    X_test_cont=X_test_cont,
                    X_test_cat=X_test_cat,
                    model=model,
                    ts_feature_names=time_series_feature_names,
                    cat_feature_names=categorical_feature_names,
                    sample_size=min(50, len(X_test_cont)),
                    background_size=min(10, len(X_test_cont))
                )
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {str(e)}")
                shap_results = {"success": False, "error": str(e)}

        # Prepare response
        response = {
            "auc": float(AUC * 100),
            "threshold": float(thresh_final),
            "metrics": metrics,
            "shap_values": {
                "time_series": shap_results.get("time_series_importances", {}),
                "categorical": shap_results.get("categorical_importances", {}),
                "top_features": shap_results.get("top_shap_features", {})
            },
            "prediction_prob":pred_probs
        }

        return safe_jsonable_encoder(response)

    except Exception as e:
        logger.exception("Prediction failed")
        raise ValueError(f"Prediction error: {str(e)}")










