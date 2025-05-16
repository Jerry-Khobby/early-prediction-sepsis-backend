from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse,ManualPredictionRequest
from app.services.predictor import predict_from_csv
from app.services.manually import ManualPredictor
from app.services.shap_explainer import get_shap_importances
import logging
from io import StringIO

import pandas as pd
import numpy as np
from tensorflow import keras 
from app.core.config import MODEL_PATH
from typing import Dict, Any
from app.services.reporter import generate_medication_suggestions,generate_clinical_explanation

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = keras.models.load_model(MODEL_PATH)

# Updated predict endpoint to directly accept CSV file
@router.post("/csv_predict")
async def predict_vitals(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file: {file.filename}")
        
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")

        # Read and validate CSV
        contents = await file.read()
        try:
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

        # Validate required columns and data types
        required = {'X_cont', 'label'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise HTTPException(status_code=400, 
                             detail=f"Missing required columns: {missing}")

        # Check X_cont contains proper array strings
        if not df['X_cont'].apply(lambda x: isinstance(x, str) and x.startswith('[')).all():
            raise HTTPException(status_code=400,
                             detail="X_cont must contain string representations of arrays")

        # Clean dataframe
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        return await predict_from_csv(df)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")




#My next route is to make do of the LLMs
@router.post("/reports/recommendations")
async def generate_report(prediction_results: dict):
    try:
        # Validate required fields
        required = {'threshold', 'metrics', 'shap_values'}
        if not required.issubset(prediction_results.keys()):
            raise ValueError(f"Missing required fields: {required - set(prediction_results.keys())}")

        # Validate metrics (only AUC and Accuracy are mandatory)
        metrics = prediction_results['metrics']
        if 'auc' not in metrics or 'accuracy' not in metrics:
            raise ValueError("Metrics must include 'auc' and 'accuracy'")

        # Generate outputs
        explanation = generate_clinical_explanation(prediction_results)
        medications = generate_medication_suggestions(prediction_results)

        return {
            "clinical": explanation,
            "medications": medications,
            "metrics_used": {
                "auc": metrics['auc'],
                "accuracy": metrics['accuracy']
            }
        }

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))






# routers/predict.py
# routers/manual.py

# Initialize during app startup
predictor = ManualPredictor(model)  # model loaded during startup

@router.post("/manual_predict")
async def manual_predict(request: ManualPredictionRequest):
    try:
        # Preprocess and predict
        X_cont, X_cat = predictor.preprocess(request)
        results = predictor.predict(X_cont, X_cat)
        
        # Ensure all values are valid before creating response
        auc_value = float(results.get("auc", 0.0) * 100) if results.get("auc") is not None else 0.0
        threshold_value = float(results.get("threshold", 0.0)) if results.get("threshold") is not None else 0.0
        prediction_value = float(results.get("prediction", 0.0)) if results.get("prediction") is not None else 0.0
        
        return {
            "auc": auc_value,
            "threshold": threshold_value,
            "shap_values": results.get("shap_values", {}),
            "prediction_probability": prediction_value
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
