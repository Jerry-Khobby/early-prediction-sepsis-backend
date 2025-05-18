import numpy as np
from typing import Dict, List, Tuple
from app.schemas.scaling import scaler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#When I mean action , I mean any first add that can be taken to help the patient 
class ManualPredictor:
    def __init__(self, model):
        self.model = model
        self.cont_features = ['HR', 'MAP', 'O2Sat', 'SBP', 'Resp']
        self.cat_features = [
            'Unit1', 'Gender', 'HospAdmTime', 'Age', 'DBP', 'Temp', 'Glucose',
            'Potassium', 'Hct', 'FiO2', 'Hgb', 'pH', 'BUN', 'WBC', 'Magnesium',
            'Creatinine', 'Platelets', 'Calcium', 'PaCO2', 'BaseExcess', 'Chloride',
            'HCO3', 'Phosphate', 'EtCO2', 'SaO2', 'PTT', 'Lactate', 'AST',
            'Alkalinephos', 'Bilirubin_total', 'TroponinI', 'Fibrinogen', 'Bilirubin_direct'
        ]
        
        # Clinical protocols configuration
        self.risk_protocols = {
            'low': {
                'interpretation': "Low probability of sepsis development within 10 hours",
                'monitoring': ["Vital signs every 4-6 hours"],
                'diagnostics': ["Consider CBC if not recent"],
                'medications': [],
                'actions': ["Document in progress notes"]
            },
            'medium': {
                'interpretation': "Moderate sepsis risk - recommend enhanced monitoring",
                'monitoring': ["Vital signs every 2 hours", "Continuous pulse oximetry"],
                'diagnostics': ["CBC", "Basic metabolic panel", "Lactate", "Blood cultures x2"],
                'medications': [
                    "IV fluids if hypotensive (Normal Saline 500-1000mL bolus)",
                    "Antipyretics PRN (Acetaminophen 650mg PO/PR every 6hr)"
                ],
                'actions': [
                    "Notify primary team",
                    "Consider sepsis alert activation"
                ]
            },
            'high': {
                'interpretation': "High sepsis probability - initiate treatment protocol",
                'monitoring': [
                    "Continuous cardiac monitoring",
                    "Hourly vital signs",
                    "Strict I/O monitoring"
                ],
                'diagnostics': [
                    "STAT blood cultures x2",
                    "Lactate",
                    "Arterial blood gas",
                    "Complete sepsis panel (CBC, CMP, PT/PTT, CRP)"
                ],
                'medications': [
                    "IV Fluids: 30mL/kg crystalloid bolus (Normal Saline/Lactated Ringers)",
                    "Broad-spectrum antibiotics:",
                    "- Community-acquired: Ceftriaxone 2g IV q24h + Azithromycin 500mg IV q24h",
                    "- Hospital-acquired: Piperacillin-Tazobactam 4.5g IV q6h",
                    "- Penicillin allergy: Ciprofloxacin 400mg IV q8h + Clindamycin 900mg IV q8h",
                    "Vasopressors if refractory hypotension (Norepinephrine 0.05-0.3 mcg/kg/min)"
                ],
                'actions': [
                    "Activate rapid response team",
                    "Prepare for ICU transfer",
                    "Notify attending physician immediately"
                ]
            },
            'critical': {
                'interpretation': "Critical sepsis risk - life-threatening condition",
                'monitoring': [
                    "Arterial line placement",
                    "Continuous ScvO2 monitoring if available",
                    "Frequent neurologic checks"
                ],
                'diagnostics': [
                    "STAT blood cultures x2 from different sites",
                    "Arterial blood gas q2h",
                    "Coagulation studies",
                    "Chest X-ray",
                    "Consider procalcitonin"
                ],
                'medications': [
                    "Aggressive fluid resuscitation (consider albumin if hypoalbuminemic)",
                    "Dual antibiotic therapy:",
                    "- Vancomycin 15-20mg/kg IV (max 2g) + Meropenem 1g IV q8h",
                    "- MRSA coverage: Add Vancomycin if risk factors present",
                    "Vasopressor therapy:",
                    "- Norepinephrine 0.05-3 mcg/kg/min (first line)",
                    "- Vasopressin 0.03 units/min (second line)",
                    "- Epinephrine 0.1-0.5 mcg/kg/min (if cardiac dysfunction)",
                    "Stress-dose steroids: Hydrocortisone 50mg IV q6h if refractory shock",
                    "Consider Drotrecogin alfa in selected cases"
                ],
                'actions': [
                    "Immediate ICU transfer",
                    "Notify critical care team STAT",
                    "Prepare for intubation if respiratory distress"
                ]
            }
        }

        self.antibiotic_options = {
            'community': [
                "Ceftriaxone 2g IV q24h + Azithromycin 500mg IV q24h",
                "Levofloxacin 750mg IV q24h"
            ],
            'hospital': [
                "Piperacillin-Tazobactam 4.5g IV q6h",
                "Meropenem 1g IV q8h",
                "Cefepime 2g IV q8h + Metronidazole 500mg IV q8h"
            ],
            'allergy': [
                "Ciprofloxacin 400mg IV q8h + Clindamycin 900mg IV q8h",
                "Aztreonam 2g IV q8h + Vancomycin (if MRSA coverage needed)"
            ]
        }

    def preprocess(self, request) -> Tuple[np.ndarray, np.ndarray]:
        """Convert manual input to model-compatible format"""
        # Process continuous features
        X_cont = np.column_stack([
            scaler.scale_continuous(request.HR, 'HR'),
            scaler.scale_continuous(request.MAP, 'MAP'),
            scaler.scale_continuous(request.O2Sat, 'O2Sat'),
            scaler.scale_continuous(request.SBP, 'SBP'),
            scaler.scale_continuous(request.Resp, 'Resp')
        ])

        # Process categorical features
        X_cat = np.array([
            scaler.scale_categorical(getattr(request, col), col) 
            for col in self.cat_features
        ], dtype=np.float32)

        return X_cont.reshape(1, 10, 5), X_cat.reshape(1, -1)

    def predict(self, X_cont: np.ndarray, X_cat: np.ndarray) -> Dict:
        """Run model prediction and generate clinical response"""
        prediction = self.model.predict([X_cont, X_cat])
        pred_prob = float(prediction[0, 1])
        risk_level = self._determine_risk_level(pred_prob)
        protocol = self.risk_protocols[risk_level]
        
        return {
            'risk_score': pred_prob,
            'risk_level': risk_level,
            'time_horizon': '10-hour sepsis risk',
            'clinical_interpretation': protocol['interpretation'],
            'key_drivers': self._get_top_features(X_cont[0], X_cat[0]),
            'clinical_protocol': {
                'monitoring': protocol['monitoring'],
                'diagnostics': protocol['diagnostics'],
                'medications': protocol['medications'],
                'actions': protocol['actions']
            },
            'antibiotic_options': self._get_antibiotic_options(risk_level),
            'clinical_warnings': self._get_clinical_warnings(risk_level),
            'disclaimer': 'Clinical recommendations require physician validation'
        }

    def _determine_risk_level(self, probability: float) -> str:
        """Categorize sepsis risk level"""
        if probability < 0.3: return 'low'
        elif probability < 0.7: return 'medium'
        elif probability < 0.9: return 'high'
        else: return 'critical'

    def _get_top_features(self, X_cont: np.ndarray, X_cat: np.ndarray) -> List[Dict]:
        """Identify top 5 contributing features"""
        # Time-series features importance
        ts_importance = {
            feat: float(np.mean(np.abs(X_cont[:, i])))
            for i, feat in enumerate(self.cont_features)
        }
        
        # Categorical features importance
        cat_importance = {
            feat: float(np.abs(val))
            for feat, val in zip(self.cat_features, X_cat[0])
        }
        
        # Combine and return top 5
        all_features = {**ts_importance, **cat_importance}
        return [
            {"feature": k, "value": v}
            for k, v in sorted(all_features.items(),
                             key=lambda x: x[1],
                             reverse=True)[:5]
        ]

    def _get_antibiotic_options(self, risk_level: str) -> Dict:
        """Return appropriate antibiotic choices based on risk"""
        if risk_level in ['low', 'medium']:
            return {'primary': self.antibiotic_options['community'][0]}
        
        return {
            'community_acquired': self.antibiotic_options['community'],
            'hospital_acquired': self.antibiotic_options['hospital'],
            'penicillin_allergy': self.antibiotic_options['allergy']
        }

    def _get_clinical_warnings(self, risk_level: str) -> List[str]:
        """Return relevant clinical warnings"""
        base_warnings = [
            "Verify medication allergies before administration",
            "Monitor for signs of anaphylaxis with first antibiotic doses"
        ]
        
        if risk_level in ['high', 'critical']:
            base_warnings.extend([
                "Assess for contraindications to fluid bolus (CHF, renal failure)",
                "Monitor for antibiotic-associated diarrhea",
                "Check renal function for dose adjustments"
            ])
        
        return base_warnings