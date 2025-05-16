# Sepsis Prediction Backend

A FastAPI-based backend service for sepsis prediction using machine learning. This service provides endpoints for predicting sepsis risk based on patient vitals, generating AI-powered medical reports, and exporting results in various formats.

## Features

- Accepts CSV uploads or manual input of patient vitals
- Runs predictions using a pre-trained ML model
- Generates explanations using SHAP values
- Creates AI-generated doctor reports using GPT-4
- Supports PDF export and email delivery of reports
- Handles PDF to CSV conversion for uploaded files

## Prerequisites

- Python 3.8+
- FastAPI
- Uvicorn
- Pandas
- NumPy
- scikit-learn
- SHAP
- OpenAI API key
- SendGrid API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sepsis_backend
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
SENDGRID_API_KEY=your_sendgrid_api_key
FROM_EMAIL=your_verified_sender@example.com
```

## Project Structure

```
sepsis_backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic services
│   └── utils/               # Utility functions
├── models/                  # ML model files
├── reports/                 # Generated reports
├── uploads/                 # Uploaded files
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## API Endpoints

### 1. Predict from File Upload
```
POST /predict/upload
```
Accepts a CSV or PDF file containing patient vitals and returns predictions.

### 2. Predict from Manual Input
```
POST /predict/manual
```
Accepts manual input of patient vitals and returns predictions.

### 3. Export Report
```
POST /report/export
```
Generates and exports a PDF report, optionally sending it via email.

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

2. Access the API documentation at:
```
http://localhost:8000/docs
```

## Model Requirements

The backend expects a pre-trained model file (`sepsis_model.pkl`) in the `models/` directory. The model should be trained to predict sepsis based on the following features:

- Unit1
- Gender
- HospAdmTime
- Age
- DBP
- Temp
- Glucose
- Potassium
- Hct
- FiO2
- Hgb
- pH
- BUN
- WBC
- Magnesium
- Creatinine
- Platelets
- Calcium
- PaCO2
- BaseExcess
- Chloride
- HCO3
- Phosphate
- EtCO2
- SaO2
- PTT
- Lactate
- AST
- Alkalinephos
- Bilirubin_total
- TroponinI
- Fibrinogen
- Bilirubin_direct
- X_cont

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
