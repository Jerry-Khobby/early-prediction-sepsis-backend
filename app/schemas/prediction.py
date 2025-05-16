"""
This file defines the Pydantic models for prediction requests and responses in the sepsis prediction API. It includes models for prediction requests and responses, including predictions, probabilities, explanations, and reports.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional,Any

class PredictionRequest(BaseModel):
    Unit1: float
    Gender: float
    HospAdmTime: float
    Age: float
    DBP: float
    Temp: float
    Glucose: float
    Potassium: float
    Hct: float
    FiO2: float
    Hgb: float
    pH: float
    BUN: float
    WBC: float
    Magnesium: float
    Creatinine: float
    Platelets: float
    Calcium: float
    PaCO2: float
    BaseExcess: float
    Chloride: float
    HCO3: float
    Phosphate: float
    EtCO2: float
    SaO2: float
    PTT: float
    Lactate: float
    AST: float
    Alkalinephos: float
    Bilirubin_total: float
    TroponinI: float
    Fibrinogen: float
    Bilirubin_direct: float
    X_cat: List[float]  # Categorical/static features
    X_cont: List[float]


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    explanations: Dict[str, float]
    report: str 
    
    
class ReportRequest(BaseModel):
    email: str =None 
    
    

class PredictionRequest(BaseModel):
    X_cat: List[List[Any]]
    X_cont: List[List[float]]
    


from pydantic import  validator
import numpy as np

class ManualPredictionRequest(BaseModel):
    # Time-series features (last 10 readings)
    HR: List[float] = Field(..., min_items=10, max_items=10)
    MAP: List[float] = Field(..., min_items=10, max_items=10)
    O2Sat: List[float] = Field(..., min_items=10, max_items=10)
    SBP: List[float] = Field(..., min_items=10, max_items=10)
    Resp: List[float] = Field(..., min_items=10, max_items=10)
    
    # Categorical features (scalar floats)
    Unit1: float = Field(..., ge=0, le=1)
    Gender: float = Field(..., ge=0, le=1)
    HospAdmTime: float = Field(..., ge=0)
    Age: float = Field(..., gt=0, le=120)
    DBP: float = Field(..., gt=0)
    Temp: float = Field(..., gt=30, lt=45)
    Glucose: float = Field(..., ge=0)
    Potassium: float = Field(..., gt=0, lt=10)
    Hct: float = Field(..., ge=0, le=100)
    FiO2: float = Field(..., ge=0.21, le=1.0)
    Hgb: float = Field(..., gt=0)
    pH: float = Field(..., gt=6.5, lt=8.0)
    BUN: float = Field(..., ge=0)
    WBC: float = Field(..., gt=0)
    Magnesium: float = Field(..., gt=0)
    Creatinine: float = Field(..., ge=0)
    Platelets: float = Field(..., ge=0)
    Calcium: float = Field(..., gt=0)
    PaCO2: float = Field(..., ge=0)
    BaseExcess: float
    Chloride: float = Field(..., gt=0)
    HCO3: float = Field(..., ge=0)
    Phosphate: float = Field(..., gt=0)
    EtCO2: float = Field(..., ge=0)
    SaO2: float = Field(..., ge=0, le=100)
    PTT: float = Field(..., gt=0)
    Lactate: float = Field(..., ge=0)
    AST: float = Field(..., ge=0)
    Alkalinephos: float = Field(..., ge=0)
    Bilirubin_total: float = Field(..., ge=0)
    TroponinI: float = Field(..., ge=0)
    Fibrinogen: float = Field(..., gt=0)
    Bilirubin_direct: float = Field(..., ge=0)

    # Handle NaNs in scalar float fields
    @validator(
        'Unit1', 'Gender', 'HospAdmTime', 'Age', 'DBP', 'Temp', 'Glucose',
        'Potassium', 'Hct', 'FiO2', 'Hgb', 'pH', 'BUN', 'WBC', 'Magnesium',
        'Creatinine', 'Platelets', 'Calcium', 'PaCO2', 'BaseExcess', 'Chloride',
        'HCO3', 'Phosphate', 'EtCO2', 'SaO2', 'PTT', 'Lactate', 'AST',
        'Alkalinephos', 'Bilirubin_total', 'TroponinI', 'Fibrinogen',
        'Bilirubin_direct',
        pre=True
    )
    def replace_nan_scalar(cls, v):
        return v if v is not None and not np.isnan(v) else np.pi

    # Handle NaNs in list fields
    @validator('HR', 'MAP', 'O2Sat', 'SBP', 'Resp', pre=True)
    def replace_nan_list(cls, v):
        return [x if x is not None and not np.isnan(x) else np.pi for x in v]