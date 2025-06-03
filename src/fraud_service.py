import bentoml
from bentoml.io import JSON
import pandas as pd
import joblib
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model with error handling
MODEL_PATH = "models/best_model.pkl"
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        logger.error(f"❌ Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    raise

# Create service object
svc = bentoml.Service("fraud_detector")

@svc.api(input=JSON(), output=JSON())
def predict(input_data: dict) -> dict:
    """
    Predict fraud for a transaction
    
    Expected input:
    {
        "amount": 150.0,
        "merchant": "Amazon",
        "location": "NewYork", 
        "card_type": "Visa"
    }
    """
    try:
        # Validate input
        required_fields = ['amount', 'merchant', 'location', 'card_type']
        for field in required_fields:
            if field not in input_data:
                return {"error": f"Missing required field: {field}", "required_fields": required_fields}
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df, columns=['merchant', 'location', 'card_type'])
        
        # Align columns with training data
        if hasattr(model, 'feature_names_in_'):
            required_columns = model.feature_names_in_
            for col in required_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded.reindex(columns=required_columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(df_encoded)[0]
        
        # Get prediction probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            prob_array = model.predict_proba(df_encoded)[0]
            probability = float(prob_array[1])  # Probability of fraud (class 1)
        
        result = {
            "prediction": int(prediction),
            "is_fraud": bool(prediction),
            "fraud_probability": probability,
            "input_data": input_data,
            "status": "success"
        }
        
        logger.info(f"✅ Prediction made: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "input_data": input_data
        }

@svc.api(input=JSON(), output=JSON())
def healthz() -> dict:
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "model_loaded": os.path.exists(MODEL_PATH),
            "service": "fraud_detector",
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@svc.api(input=JSON(), output=JSON())
def info() -> dict:
    """Get model information"""
    try:
        model_info = {
            "model_type": str(type(model).__name__),
            "features": list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else "Unknown",
            "model_path": MODEL_PATH
        }
        
        return {
            "status": "success",
            "model_info": model_info
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
