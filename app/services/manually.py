import pandas as pd
import numpy as np
from app.core.config import MODEL_PATH,MEAN_STD_SCALING_PATH
from tensorflow.keras.utils import to_categorical
from fastapi.encoders import jsonable_encoder
from io import StringIO
from tensorflow import keras 
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.aur_roc import roc
from app.schemas.evaluation import evaluate_model
from app.services.shap_explainer import get_shap_importances
from fastapi import UploadFile
from app.schemas.prediction import ManualPredictionRequest
import logging
import ast 
import shap
import pickle
from app.schemas.safe_encoder import safe_jsonable_encoder,clean_json
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model = keras.models.load_model(MODEL_PATH)

#I want to open the mean_std_scaling.pkl file and load the contents into a dictionary
with open(MEAN_STD_SCALING_PATH, 'rb') as f:
    df_mean_std = pickle.load(f)




class ManualPredictor:
    def __init__(self, model):
        self.model = model
        self.cont_features = ['HR', 'MAP', 'O2Sat', 'SBP', 'Resp']
        self.cols_to_bin = [
            'Unit1', 'Gender', 'HospAdmTime', 'Age', 'DBP', 'Temp', 'Glucose',
            'Potassium', 'Hct', 'FiO2', 'Hgb', 'pH', 'BUN', 'WBC', 'Magnesium',
            'Creatinine', 'Platelets', 'Calcium', 'PaCO2', 'BaseExcess', 'Chloride',
            'HCO3', 'Phosphate', 'EtCO2', 'SaO2', 'PTT', 'Lactate', 'AST',
            'Alkalinephos', 'Bilirubin_total', 'TroponinI', 'Fibrinogen', 'Bilirubin_direct'
        ]
        self.cols_series=['X_cont_1', 'X_cont_2', 'X_cont_3', 'X_cont_4', 'X_cont_5']

    def preprocess(self, request):
        """Convert manual input to model-compatible format"""
        # Process continuous features
        X_cont = np.array([
            request.HR[-10:],
            request.MAP[-10:],
            request.O2Sat[-10:],
            request.SBP[-10:],
            request.Resp[-10:]
        ]).T  # Shape: (10, 5)

        # Standardize
        for i, col in enumerate(self.cont_features):
            X_cont[:, i] = (X_cont[:, i] - df_mean_std[col]['mean']) / df_mean_std[col]['std']

        # Process categorical features
        cat_values = [getattr(request, col) for col in self.cols_to_bin]
        for i, col in enumerate(self.cols_to_bin):
            if col not in ['Gender', 'Unit1']:
                cat_values[i] = (cat_values[i] - df_mean_std[col]['mean']) / df_mean_std[col]['std']

        X_cat = np.array([cat_values], dtype=np.float32)
        return X_cont.reshape(1, 10, 5), X_cat

    def predict(self, X_cont, X_cat):
        """Run model prediction and generate metrics"""
        prediction = self.model.predict([X_cont, X_cat])
        mock_label = np.array([[1]])
        mock_label_cat = to_categorical(mock_label)
        
        _, threshold, auc = roc(prediction[:, 1], mock_label_cat[:, 1], 'manual')
        shap_values = get_shap_importances(
            X_test_cont=X_cont,
            X_test_cat=X_cat, 
            model=self.model,
            sample_size=1,
            background_size=1,
            cat_feature_names=self.cols_to_bin,
            ts_feature_names=self.cols_series,
          
        )
        auc_percentage = float(auc * 100) if auc is not None else 0.0
        prediction_value = float(prediction[0, 1]) if prediction is not None else 0.0
        threshold_value = float(threshold) if threshold is not None else 0.0
        auc_value = float(auc) if auc is not None else 0.0
        
        response= {
          "auc_percentage":auc_percentage,
            "prediction": prediction_value,
            "threshold": threshold_value,
            "auc": auc_value,
            "shap_values":{
                "time_series_importances": shap_values.get("time_series_importances", {}),
                "categorical_importances": shap_values.get("categorical_importances", {}),
                "top_shap_features": shap_values.get("top_shap_features", {})
            }
        }
        cleaned_response=clean_json(response)
        return safe_jsonable_encoder(cleaned_response)