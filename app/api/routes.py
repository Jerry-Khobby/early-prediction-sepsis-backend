from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse,ManualPredictionRequest
from app.services.predictor import predict_from_csv
from app.services.manually import ManualPredictor
from app.services.shap_explainer import get_shap_importances
import logging
from io import StringIO
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pydantic import BaseModel
import pandas as pd
from tensorflow import keras 
from app.core.config import MODEL_PATH
from app.services.reporter import generate_medication_suggestions,generate_clinical_explanation
from dotenv import load_dotenv
load_dotenv()


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

predictor = ManualPredictor(model)

@router.post("/manual_predict")
async def manual_predict(request: ManualPredictionRequest):
    try:
        # Preprocess and predict
        X_cont, X_cat = predictor.preprocess(request)
        results = predictor.predict(X_cont, X_cat)
        
        return {
            "risk_assessment": {
                "score": results['risk_score'],
                "level": results['risk_level'],
                "time_frame": results['time_horizon'],
                "interpretation": results['clinical_interpretation'],
                "detailed_analysis": results['clinical_analysis']
            },
            "key_risk_factors": results['key_drivers'],
            "clinical_guidance": {
                "monitoring": results['clinical_protocol']['monitoring'],
                "diagnostic_tests": results['clinical_protocol']['diagnostics'],
                "treatment_options": {
                    "immediate_medications": results['clinical_protocol']['medications'],
                    "antibiotic_choices": results['antibiotic_options']
                },
                "required_actions": results['clinical_protocol']['actions']
            },
            "safety_alerts": results['clinical_warnings'],
            "disclaimers": [
                results['disclaimer'],
                "All medication doses require validation for renal/hepatic function",
                "Actual treatment may vary based on patient-specific factors"
            ]
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "clinical_advice": [
                    "While system is unavailable, follow standard sepsis protocols:",
                    "1. Obtain blood cultures",
                    "2. Administer broad-spectrum antibiotics",
                    "3. Initiate fluid resuscitation",
                    "4. Contact senior clinician immediately"
                ]
            }
        )







class EmailRequest(BaseModel):
    email: str
    pdf_base64: str
    report_name: str

@router.post("/send-report")
async def send_report(request: EmailRequest):
    try:
        # Email configuration from environment variables
        email_user = os.getenv("EMAIL_USER")
        email_password = os.getenv("EMAIL_PASSWORD")
        email_host = os.getenv("EMAIL_HOST", "smtp.gmail.com")
        email_port = int(os.getenv("EMAIL_PORT", 587))
        
        if not email_user or not email_password:
            raise HTTPException(status_code=500, detail="Email service not configured")

        # Decode the base64 PDF
        pdf_bytes = base64.b64decode(request.pdf_base64)

        # Create email message
        msg = MIMEMultipart()
        msg['From'] = f"MedicalAI Reports <{email_user}>"
        msg['To'] = request.email
        msg['Subject'] = f"Your {request.report_name} Report"

        # Email body
        body = f"Please find attached your {request.report_name} report."
        msg.attach(MIMEText(body, 'plain'))

        # Attach PDF
        part = MIMEApplication(pdf_bytes, Name=f"{request.report_name}.pdf")
        part['Content-Disposition'] = f'attachment; filename="{request.report_name}.pdf"'
        msg.attach(part)

        # Send email
        with smtplib.SMTP(email_host, email_port) as server:
            server.starttls()
            server.login(email_user, email_password)
            server.send_message(msg)

        return {"status": "success", "message": "Email sent successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")