from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd

def safe_jsonable_encoder(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # Fallback for any other NumPy scalar
        return obj.item()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_jsonable_encoder(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_jsonable_encoder(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return safe_jsonable_encoder(vars(obj))
    return jsonable_encoder(obj)





import math

def clean_json(obj):
    """Recursively replace NaN and inf values in dicts/lists with None."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(x) for x in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj
    
    
    
def safe_jsonable_encoder_second(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_jsonable_encoder(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_jsonable_encoder(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return safe_jsonable_encoder(vars(obj))
    return obj