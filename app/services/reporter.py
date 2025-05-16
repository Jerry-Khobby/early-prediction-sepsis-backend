from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client (add your API key)
client = OpenAI(api_key="your-api-key-here")  # Replace with your actual API key

def generate_medication_suggestions(prediction_results: dict, model_version: str = "gpt-4") -> dict:
    """
    Generates targeted medication recommendations based on sepsis prediction results.
    Requires initialized OpenAI client.
    """
    try:
        if not client:
            raise ValueError("OpenAI client not initialized")
            
        # Extract prediction data
        probability = prediction_results.get('threshold', 0)
        top_features = prediction_results.get('shap_values', {}).get('top_features', {})
        
        # Prepare clinical context
        top_5_markers = dict(sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:5])
        markers_context = "\n".join([f"- {marker} (importance: {importance:.3f})" 
                                   for marker, importance in top_5_markers.items()])
        
        prompt = f"""
        **Clinical Scenario**:
        Patient with {probability:.2%} sepsis risk probability.
        Key risk markers:
        {markers_context}

        **Request**:
        Provide specific medication recommendations that:
        1. Target sepsis prevention
        2. Follow current guidelines
        3. Include typical adult dosing
        4. Categorize by urgency

        **Format**:
        [Immediate Interventions]
        - Medication 1: Dose, frequency, route
        [Preventive Measures]
        - Medication 2: Dose, frequency, route
        """
        
        response = client.chat.completions.create(
            model=model_version,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2
        )
        
        return {
            "medications": response.choices[0].message.content.strip(),
            "critical_features": top_5_markers,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Medication suggestion failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "recommendation": "Check API key and connection"
        }

def generate_clinical_explanation(prediction_results: dict, model_version: str = "gpt-4") -> dict:
    """
    Generates clinical interpretation and action recommendations.
    Requires initialized OpenAI client.
    """
    try:
        if not client:
            raise ValueError("OpenAI client not initialized")
            
        probability = prediction_results.get('threshold', 0)
        top_features = prediction_results.get('shap_values', {}).get('top_features', {})
        top_3_markers = dict(sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:3])
        
        prompt = f"""
        Patient with {probability:.2%} sepsis risk.
        Top risk factors: {", ".join(top_3_markers.keys())}
        
        Provide:
        1. Plain-language interpretation
        2. 3-5 prioritized actions
        3. Modifiable risk factors
        4. Monitoring timeline
        """
        
        response = client.chat.completions.create(
            model=model_version,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.3
        )
        
        return {
            "explanation": response.choices[0].message.content.strip(),
            "risk_factors": top_3_markers,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Clinical explanation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "recommendation": "Verify inputs and API connection"
        }