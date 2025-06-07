# src/fraud_service.py
# Simple fraud detection service without classes

import bentoml
from bentoml.io import JSON
import pandas as pd
import joblib
import logging
import os
import json
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
METADATA_PATH = "models/metadata.json"
model = None
feature_columns = None
metadata = {}

def load_model():
    """Load the trained model"""
    global model, feature_columns, metadata
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            
            # Try to get feature names
            if hasattr(model, 'feature_names_in_'):
                feature_columns = list(model.feature_names_in_)
            else:
                # Fallback feature columns
                feature_columns = [
                    'amount', 'merchant_Amazon', 'merchant_BestBuy', 'merchant_Target',
                    'merchant_Walmart', 'merchant_eBay', 'location_NewYork', 
                    'location_LosAngeles', 'card_type_Visa', 'card_type_MasterCard', 
                    'card_type_Amex'
                ]
            
            logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
            logger.info(f"📊 Features: {len(feature_columns)} total")
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        # Load metadata if available
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            logger.info("✅ Model metadata loaded")
        else:
            metadata = {
                "model_type": str(type(model).__name__) if model else "Unknown",
                "version": "1.0.0"
            }
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

def prepare_features(input_data):
    """Prepare and encode features for prediction"""
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['merchant', 'location', 'card_type'])
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match training data
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    
    return df_encoded

def make_prediction(input_data):
    """Make fraud prediction"""
    try:
        # Prepare features
        features_df = prepare_features(input_data)
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Get prediction probability if available
        probability = None
        confidence = "medium"
        
        if hasattr(model, 'predict_proba'):
            prob_array = model.predict_proba(features_df)[0]
            probability = float(prob_array[1])  # Probability of fraud (class 1)
            
            # Determine confidence level
            if probability > 0.8 or probability < 0.2:
                confidence = "high"
            elif probability > 0.6 or probability < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
        
        result = {
            "prediction": int(prediction),
            "is_fraud": bool(prediction),
            "fraud_probability": probability,
            "confidence": confidence,
            "input_data": input_data,
            "model_info": {
                "model_type": metadata.get("model_type", "Unknown"),
                "version": metadata.get("version", "1.0.0")
            },
            "status": "success"
        }
        
        logger.info(f"✅ Prediction made: fraud={bool(prediction)}, prob={probability:.3f if probability else 'N/A'}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "input_data": input_data
        }

# Load model on startup
load_model()

# Create BentoML service
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
    # Validate input
    required_fields = ['amount', 'merchant', 'location', 'card_type']
    for field in required_fields:
        if field not in input_data:
            return {
                "error": f"Missing required field: {field}", 
                "required_fields": required_fields,
                "status": "validation_error"
            }
    
    # Validate data types
    try:
        input_data['amount'] = float(input_data['amount'])
    except (ValueError, TypeError):
        return {
            "error": "Amount must be a valid number",
            "status": "validation_error"
        }
    
    # Make prediction
    return make_prediction(input_data)

@svc.api(input=JSON(), output=JSON())
def healthz() -> dict:
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "fraud_detector",
            "version": metadata.get("version", "1.0.0"),
            "model_loaded": model is not None,
            "model_type": metadata.get("model_type", "Unknown"),
            "features_count": len(feature_columns) if feature_columns else 0,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@svc.api(input=JSON(), output=JSON())
def info() -> dict:
    """Get detailed model information"""
    try:
        return {
            "status": "success",
            "service_info": {
                "name": "fraud_detector",
                "version": metadata.get("version", "1.0.0"),
                "description": "Fraud detection service"
            },
            "model_info": {
                "model_type": metadata.get("model_type", "Unknown"),
                "model_path": MODEL_PATH,
                "features": feature_columns,
                "features_count": len(feature_columns) if feature_columns else 0
            },
            "sample_request": {
                "amount": 150.0,
                "merchant": "Amazon",
                "location": "NewYork",
                "card_type": "Visa"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
