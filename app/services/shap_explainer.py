import shap
import numpy as np
import joblib
from openai import OpenAI 
from app.core.config import OPENAI_API_KEY
import pandas as pd
import logging
from tqdm import tqdm 
from typing import Union, List, Dict, Optional
# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)



logger = logging.getLogger(__name__)

def get_shap_importances(
    X_test_cont: np.ndarray,
    X_test_cat: np.ndarray,
    model,
    ts_feature_names: List[str] = None,
    cat_feature_names: List[str] = None,
    sample_size: int = 50,
    background_size: int = 10
) -> Dict:
    """
    Robust SHAP explainer that handles time-series and categorical features separately.
    
    Args:
        X_test_cont: 3D array of continuous time-series features (samples, timesteps, features)
        X_test_cat: 2D array of categorical features (samples, features)
        model: The trained dual-input model
        ts_feature_names: Names for time-series features (must match X_test_cont.shape[2])
        cat_feature_names: Names for categorical features (must match X_test_cat.shape[1])
        sample_size: Number of samples to explain
        background_size: Number of samples for background distribution
    
    Returns:
        Dictionary containing SHAP explanations and feature importances
    """
    try:
        # ===== 1. Input Validation =====
        if len(X_test_cont.shape) != 3:
            raise ValueError(f"X_test_cont must be 3D, got {X_test_cont.shape}")
        if len(X_test_cat.shape) != 2:
            raise ValueError(f"X_test_cat must be 2D, got {X_test_cat.shape}")

        n_ts_features = X_test_cont.shape[2]
        n_cat_features = X_test_cat.shape[1]

        # ===== 2. Data Sampling =====
        background_cont = X_test_cont[:background_size]
        background_cat = X_test_cat[:background_size]
        explanation_cont = X_test_cont[:sample_size]
        explanation_cat = X_test_cat[:sample_size]

        # ===== 3. SHAP Explanation =====
        logger.info("Using GradientExplainer for dual-input model...")
        explainer = shap.GradientExplainer(
            model,
            [background_cont, background_cat]
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(
            [explanation_cont, explanation_cat],
            nsamples=50
        )

        # Handle multi-output models
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_cont, shap_cat = shap_values
            logger.info(f"shap_cont shape: {np.shape(shap_cont)}")
            logger.info(f"shap_cat shape: {np.shape(shap_cat)}")
        else:
            raise ValueError(f"Expected 2 SHAP arrays, got {len(shap_values)}")

        # ===== 4. Result Processing =====
        # Calculate mean absolute SHAP values
        ts_importance = np.abs(shap_cont).mean(axis=(0, 1)).mean(axis=-1)  # shape: (5,)
        cat_importance = np.abs(shap_cat).mean(axis=0).mean(axis=-1)  # shape: (33,)

        # Generate default names if not provided
        ts_names = ts_feature_names or [f"TimeSeries_{i}" for i in range(n_ts_features)]
        cat_names = cat_feature_names or [f"Categorical_{i}" for i in range(n_cat_features)]

        # Validate name lengths
        if len(ts_names) != n_ts_features:
            raise ValueError(f"Expected {n_ts_features} time-series names, got {len(ts_names)}")
        if len(cat_names) != n_cat_features:
            raise ValueError(f"Expected {n_cat_features} categorical names, got {len(cat_names)}")

        # Create importance dictionaries
        ts_importances = dict(zip(ts_names, ts_importance.tolist()))
        cat_importances = dict(zip(cat_names, cat_importance.tolist()))

        # Combine for top features
        all_importances = {**ts_importances, **cat_importances}
        top_features = dict(sorted(all_importances.items(), key=lambda x: x[1], reverse=True)[:10])

        return {
            "success": True,
            "time_series_importances": ts_importances,
            "categorical_importances": cat_importances,
            "top_shap_features": top_features,
            "time_series_shap": shap_cont.tolist(),
            "categorical_shap": shap_cat.tolist()
        }

    except Exception as e:
        logger.exception("SHAP explanation failed")
        return {
            "success": False,
            "error": str(e),
            "recommendation": "Check input shapes and feature names"
        }

