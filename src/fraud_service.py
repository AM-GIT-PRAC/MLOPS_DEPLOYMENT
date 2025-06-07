import bentoml
from bentoml.io import JSON
import pandas as pd
import joblib
import logging
import os
import json
import numpy as np
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
METADATA_PATH = "models/metadata.json"

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.metadata = {}
        self.load_model()
        self.load_metadata()
    
    def load_model(self):
        """Load the trained model with error handling"""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                
                # Try to get feature names
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_columns = list(self.model.feature_names_in_)
                else:
                    # Fallback feature columns based on expected structure
                    self.feature_columns = [
                        'amount', 'merchant_Amazon', 'merchant_BestBuy', 'merchant_Target',
                        'merchant_Walmart', 'merchant_eBay', 'location_NewYork', 
                        'location_LosAngeles', 'card_type_Visa', 'card_type_MasterCard', 
                        'card_type_Amex'
                    ]
                
                logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
                logger.info(f"📊 Features ({len(self.feature_columns)}): {self.feature_columns[:5]}...")
            else:
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def load_metadata(self):
        """Load model metadata if available"""
        try:
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("✅ Model metadata loaded")
            else:
                self.metadata = {
                    "model_type": str(type(self.model).__name__) if self.model else "Unknown",
                    "version": "1.0.0",
                    "description": "Fraud detection model"
                }
        except Exception as e:
            logger.warning(f"⚠️ Could not load metadata: {e}")
            self.metadata = {}
    
    def prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare and encode features for prediction"""
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df, columns=['merchant', 'location', 'card_type'])
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder columns to match training data
        df_encoded = df_encoded.reindex(columns=self.feature_columns, fill_value=0)
        
        return df_encoded
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make fraud prediction"""
        try:
            # Prepare features
            features_df = self.prepare_features(input_data)
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            
            # Get prediction probability if available
            probability = None
            confidence = "medium"
            
            if hasattr(self.model, 'predict_proba'):
                prob_array = self.model.predict_proba(features_df)[0]
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
                    "model_type": self.metadata.get("model_type", "Unknown"),
                    "version": self.metadata.get("version", "1.0.0")
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

# Initialize model
fraud_model = FraudDetectionModel()

# Create BentoML service
svc = bentoml.Service("fraud_detector", runners=[])

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
    return fraud_model.predict(input_data)

@svc.api(input=JSON(), output=JSON())
def healthz() -> dict:
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "fraud_detector",
            "version": fraud_model.metadata.get("version", "1.0.0"),
            "model_loaded": fraud_model.model is not None,
            "model_type": fraud_model.metadata.get("model_type", "Unknown"),
            "features_count": len(fraud_model.feature_columns) if fraud_model.feature_columns else 0,
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
                "version": fraud_model.metadata.get("version", "1.0.0"),
                "description": fraud_model.metadata.get("description", "Fraud detection service")
            },
            "model_info": {
                "model_type": fraud_model.metadata.get("model_type", "Unknown"),
                "model_path": MODEL_PATH,
                "features": fraud_model.feature_columns,
                "features_count": len(fraud_model.feature_columns) if fraud_model.feature_columns else 0
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

@svc.api(input=JSON(), output=JSON())
def batch_predict(input_data: dict) -> dict:
    """
    Batch prediction for multiple transactions
    
    Expected input:
    {
        "transactions": [
            {"amount": 150.0, "merchant": "Amazon", "location": "NewYork", "card_type": "Visa"},
            {"amount": 50.0, "merchant": "Walmart", "location": "Chicago", "card_type": "MasterCard"}
        ]
    }
    """
    try:
        if "transactions" not in input_data:
            return {"error": "Missing 'transactions' field", "status": "validation_error"}
        
        transactions = input_data["transactions"]
        if not isinstance(transactions, list):
            return {"error": "Transactions must be a list", "status": "validation_error"}
        
        results = []
        for i, transaction in enumerate(transactions):
            result = fraud_model.predict(transaction)
            result["transaction_id"] = i
            results.append(result)
        
        return {
            "status": "success",
            "batch_size": len(transactions),
            "results": results,
            "summary": {
                "total_fraud": sum(1 for r in results if r.get("is_fraud", False)),
                "total_legitimate": sum(1 for r in results if not r.get("is_fraud", True))
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
