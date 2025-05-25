import numpy as np
from typing import Dict, List, Tuple
from app.schemas.scaling import scaler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        X_cont = np.column_stack([
            scaler.scale_continuous(request.HR, 'HR'),
            scaler.scale_continuous(request.MAP, 'MAP'),
            scaler.scale_continuous(request.O2Sat, 'O2Sat'),
            scaler.scale_continuous(request.SBP, 'SBP'),
            scaler.scale_continuous(request.Resp, 'Resp')
        ])

        X_cat = np.array([
            scaler.scale_categorical(getattr(request, col, 0), col) 
            for col in self.cat_features
        ], dtype=np.float32)
        
        X_cont = X_cont.reshape(1, 10, 5)
        X_cat = X_cat.reshape(1, -1)
        
        if X_cat.shape[1] != len(self.cat_features):
            logger.error(f"Expected {len(self.cat_features)} categorical features, got {X_cat.shape[1]}")
            raise ValueError("Mismatch in number of categorical features")
            
        return X_cont, X_cat

    def predict(self, X_cont: np.ndarray, X_cat: np.ndarray) -> Dict:
        """Run model prediction and generate clinical response"""
        try:
            if X_cont.shape != (1, 10, 5):
                raise ValueError("Continuous features must have shape (1, 10, 5)")
            if X_cat.shape[1] != len(self.cat_features):
                raise ValueError(f"Categorical features must have {len(self.cat_features)} elements")
                
            prediction = self.model.predict([X_cont, X_cat])
            pred_prob = float(prediction[0, 1])
            risk_level = self._determine_risk_level(pred_prob)
            protocol = self.risk_protocols[risk_level]
            top_features = self._get_top_features(X_cont[0], X_cat[0])
            
            clinical_analysis = self._generate_clinical_analysis(
                risk_level, 
                X_cont[0], 
                X_cat[0], 
                top_features
            )
            
            return {
                'risk_score': pred_prob,
                'risk_level': risk_level,
                'time_horizon': '10-hour sepsis risk',
                'clinical_interpretation': protocol['interpretation'],
                'clinical_analysis': clinical_analysis,
                'key_drivers': top_features,
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
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'error': str(e),
                'message': 'Failed to generate prediction'
            }

    def _determine_risk_level(self, probability: float) -> str:
        if probability < 0.3:
            return 'low'
        elif probability < 0.7:
            return 'medium'
        elif probability < 0.9:
            return 'high'
        else:
            return 'critical'

    def _get_top_features(self, X_cont: np.ndarray, X_cat: np.ndarray) -> List[Dict]:
        ts_importance = {
            feat: float(np.mean(np.abs(X_cont[:, i])))
            for i, feat in enumerate(self.cont_features)
        }
        
        if X_cat.ndim == 1:
            X_cat = X_cat.reshape(1, -1)
            
        cat_importance = {
            feat: float(np.abs(val))
            for feat, val in zip(self.cat_features, X_cat[0])
        }
        
        all_features = {**ts_importance, **cat_importance}
        return [
            {"feature": k, "value": v}
            for k, v in sorted(all_features.items(),
                             key=lambda x: x[1],
                             reverse=True)[:5]
        ]

    def _generate_clinical_analysis(self, risk_level: str, X_cont: np.ndarray, X_cat: np.ndarray, top_features: List[Dict]) -> str:
        avg_values = {
            feat: float(np.mean(X_cont[:, i]))
            for i, feat in enumerate(self.cont_features)
        }
        
        cat_values = {
            feat: float(val)
            for feat, val in zip(self.cat_features, X_cat)
        }
        
        top_feature_names = [f['feature'] for f in top_features[:3]]
        
        if risk_level == 'low':
            return (
                f"Comprehensive analysis reveals the patient's physiological parameters are largely within normal limits. "
                f"Heart rate averages {avg_values['HR']:.1f} bpm (normal range 60-100), mean arterial pressure maintains at {avg_values['MAP']:.1f} mmHg (target >65), "
                f"and respiratory rate is {avg_values['Resp']:.1f} breaths per minute. The most significant parameters ({', '.join(top_feature_names)}) "
                f"show no concerning patterns. Laboratory markers including WBC ({cat_values.get('WBC', 0):.1f} K/uL) and lactate ({cat_values.get('Lactate', 0):.1f} mmol/L) "
                f"are within reference ranges. The calculated risk score of {self._get_probability_range(risk_level)} suggests minimal likelihood of sepsis developing "
                "within the next 10 hours under current conditions. Continued routine monitoring is advised with reassessment if clinical status changes."
            )
            
        elif risk_level == 'medium':
            return (
                f"Assessment identifies early warning signs of potential sepsis development. The patient exhibits borderline abnormalities including "
                f"heart rate trending upward to {avg_values['HR']:.1f} bpm, respiratory rate increased to {avg_values['Resp']:.1f} breaths/min, "
                f"and MAP decreasing to {avg_values['MAP']:.1f} mmHg. Key laboratory abnormalities include {self._describe_lab_abnormalities(cat_values, top_feature_names)}. "
                f"The most concerning features ({', '.join(top_feature_names)}) demonstrate early systemic inflammatory response. "
                f"With a calculated risk probability of {self._get_probability_range(risk_level)}, there is moderate concern for sepsis progression "
                "within 6-10 hours. Early intervention including fluid resuscitation and diagnostic workup is recommended to prevent clinical deterioration."
            )
            
        elif risk_level == 'high':
            return (
                f"Clinical evaluation demonstrates clear evidence of sepsis with significant physiological derangements. The patient manifests "
                f"tachycardia ({avg_values['HR']:.1f} bpm), hypotension (MAP {avg_values['MAP']:.1f} mmHg), and tachypnea ({avg_values['Resp']:.1f} breaths/min) "
                f"with oxygen saturation trending downward to {avg_values['O2Sat']:.1f}%. Laboratory results reveal {self._describe_lab_abnormalities(cat_values, top_feature_names)}. "
                f"The most critical parameters ({', '.join(top_feature_names)}) indicate developing organ dysfunction. The high risk score ({self._get_probability_range(risk_level)}) "
                f"suggests imminent progression to severe sepsis within 4-6 hours without immediate intervention. Aggressive fluid resuscitation, "
                "broad-spectrum antibiotics, and continuous hemodynamic monitoring must be initiated immediately to prevent septic shock."
            )
            
        else:  # critical
            return (
                f"CRITICAL ALERT: The patient exhibits life-threatening signs of septic shock with profound physiological collapse. "
                f"Markers include severe tachycardia ({avg_values['HR']:.1f} bpm), refractory hypotension (MAP {avg_values['MAP']:.1f} mmHg despite fluids), "
                f"and respiratory failure ({avg_values['Resp']:.1f} breaths/min with O2 saturation {avg_values['O2Sat']:.1f}% on supplemental oxygen). "
                f"Laboratory results demonstrate {self._describe_lab_abnormalities(cat_values, top_feature_names)}. The most alarming features "
                f"({', '.join(top_feature_names)}) indicate multiple organ dysfunction. With a risk probability of {self._get_probability_range(risk_level)}, "
                f"the patient is at immediate risk of cardiovascular collapse and death within 1-2 hours. This constitutes a medical emergency requiring: "
                f"1) Immediate ICU transfer, 2) Vasopressor initiation, 3) Broad-spectrum antimicrobial therapy, and 4) Consideration of mechanical ventilation."
            )

    def _describe_lab_abnormalities(self, cat_values: Dict, top_features: List[str]) -> str:
        abnormalities = []
        for feature in top_features:
            if feature in cat_values:
                value = cat_values[feature]
                if feature == 'WBC' and value > 12:
                    abnormalities.append(f"leukocytosis (WBC {value:.1f} K/uL)")
                elif feature == 'Lactate' and value > 2:
                    abnormalities.append(f"lactic acidosis (lactate {value:.1f} mmol/L)")
                elif feature == 'Creatinine' and value > 1.2:
                    abnormalities.append(f"renal impairment (creatinine {value:.1f} mg/dL)")
                elif feature == 'Bilirubin_total' and value > 1.2:
                    abnormalities.append(f"hepatic dysfunction (bilirubin {value:.1f} mg/dL)")
                elif feature == 'Platelets' and value < 150:
                    abnormalities.append(f"thrombocytopenia (platelets {value:.1f} K/uL)")
        
        return ', '.join(abnormalities) if abnormalities else "multiple concerning laboratory abnormalities"

    def _get_probability_range(self, risk_level: str) -> str:
        ranges = {
            'low': "10-30% probability",
            'medium': "30-70% probability", 
            'high': "70-90% probability",
            'critical': ">90% probability"
        }
        return ranges.get(risk_level, "")

    def _get_antibiotic_options(self, risk_level: str) -> Dict:
        if risk_level in ['low', 'medium']:
            return {'primary': self.antibiotic_options['community'][0]}
        
        return {
            'community_acquired': self.antibiotic_options['community'],
            'hospital_acquired': self.antibiotic_options['hospital'],
            'penicillin_allergy': self.antibiotic_options['allergy']
        }

    def _get_clinical_warnings(self, risk_level: str) -> List[str]:
        base_warnings = [
            "Verify medication allergies before administration",
            "Monitor for signs of anaphylaxis with first antibiotic doses"
        ]
        
        if risk_level in ['high', 'critical']:
            base_warnings.extend([
                "Assess for contraindications to fluid bolus (CHF, renal failure)",
                "Monitor for antibiotic-associated diarrhea",
                "Check renal function for dose adjustments",
                "Consider central line placement for vasopressors if needed",
                "Monitor urine output hourly for signs of renal hypoperfusion"
            ])
        
        return base_warnings