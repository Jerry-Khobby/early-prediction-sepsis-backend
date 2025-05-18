import os
import numpy as np
import pandas as pd
from typing import Dict
from app.core.config import MEAN_STD_SCALING_PATH
import pickle
import logging

logger = logging.getLogger(__name__)

class FeatureScaler:
    def __init__(self):
        self.params = self._load_params()
        self._validate_params()
        
    def _load_params(self) -> Dict:
        """Load and convert scaling parameters from pickle file"""
        if not os.path.exists(MEAN_STD_SCALING_PATH):
            raise FileNotFoundError(
                f"Scaling parameters file not found at: {MEAN_STD_SCALING_PATH}"
            )
        
        try:
            with open(MEAN_STD_SCALING_PATH, 'rb') as f:
                df_stats = pickle.load(f)
                
            # Convert DataFrame to expected dictionary format
            params = {
                'mean': df_stats.loc['mean'].to_dict(),
                'std': df_stats.loc['std'].to_dict()
            }
            
            return params
        except Exception as e:
            logger.error(f"Failed to load scaling parameters: {str(e)}")
            raise

    def _validate_params(self):
        """Validate that all required features exist in scaling parameters"""
        required_features = [
            'HR', 'MAP', 'O2Sat', 'SBP', 'Resp',  # Continuous features
            'Age', 'DBP', 'Temp', 'Glucose', 'Potassium', 'Hct', 'FiO2', 
            'Hgb', 'pH', 'BUN', 'WBC', 'Magnesium', 'Creatinine', 'Platelets',
            'Calcium', 'PaCO2', 'BaseExcess', 'Chloride', 'HCO3', 'Phosphate',
            'EtCO2', 'SaO2', 'PTT', 'Lactate', 'AST', 'Alkalinephos',
            'Bilirubin_total', 'TroponinI', 'Fibrinogen', 'Bilirubin_direct'
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in self.params['mean'] or feature not in self.params['std']:
                missing_features.append(feature)
                
        if missing_features:
            logger.warning(f"Missing scaling parameters for features: {missing_features}")
            # Initialize default values for missing features
            for feature in missing_features:
                self.params['mean'][feature] = 0
                self.params['std'][feature] = 1
    
    def scale_continuous(self, values: list, feature: str) -> np.ndarray:
        """Scale continuous features using stored mean/std"""
        try:
            mean = self.params['mean'][feature]
            std = max(self.params['std'][feature], 1e-6)  # Avoid division by zero
            return (np.array(values) - mean) / std
        except KeyError:
            logger.warning(f"Using default scaling for feature {feature}")
            return np.array(values)  # Return unscaled if feature missing
        except Exception as e:
            logger.error(f"Error scaling feature {feature}: {str(e)}")
            raise ValueError(f"Error scaling feature {feature}: {str(e)}")
    
    def scale_categorical(self, value: float, feature: str) -> float:
        """Scale categorical features (except binary ones)"""
        if feature in ['Gender', 'Unit1']:
            return value
        try:
            mean = self.params['mean'][feature]
            std = max(self.params['std'][feature], 1e-6)
            return (value - mean) / std
        except KeyError:
            logger.warning(f"Using default scaling for categorical feature {feature}")
            return value  # Return unscaled if feature missing
        except Exception as e:
            logger.error(f"Error scaling categorical feature {feature}: {str(e)}")
            raise ValueError(f"Error scaling categorical feature {feature}: {str(e)}")

# Singleton instance
scaler = FeatureScaler()