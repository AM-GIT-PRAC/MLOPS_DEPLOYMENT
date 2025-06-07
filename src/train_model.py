# src/train_model.py
# Training with static pre-generated data

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, roc_auc_score)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'fraud-detection-production')
DOCKER_IMAGE_TAG = os.getenv('DOCKER_IMAGE_TAG', f'v{datetime.now().strftime("%Y%m%d-%H%M%S")}')

def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            print(f"✅ Created experiment: {MLFLOW_EXPERIMENT_NAME}")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using experiment: {MLFLOW_EXPERIMENT_NAME}")
        
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        return experiment_id
    except Exception as e:
        print(f"⚠️ MLflow setup: {e}")
        return None

def load_static_data():
    """Load pre-generated static data from GitHub"""
    print("📊 Loading static training data...")
    
    # Check if static data exists
    if not os.path.exists("data/transactions.csv"):
        raise FileNotFoundError(
            "❌ Static data not found!\n"
            "Please generate data first:\n"
            "  python3 src/generate_data.py\n"
            "  git add data/\n"
            "  git commit -m 'Add training data'\n"
            "  git push origin MLOPS_Change_2"
        )
    
    # Load data
    data = pd.read_csv("data/transactions.csv")
    print(f"📈 Loaded {len(data)} static transactions")
    
    # Load metadata if available
    metadata = {}
    if os.path.exists("data/dataset_metadata.json"):
        with open("data/dataset_metadata.json", 'r') as f:
            metadata = json.load(f)
        print("✅ Dataset metadata loaded")
    
    # Load feature mapping
    feature_mapping = {}
    if os.path.exists("data/feature_mapping.json"):
        with open("data/feature_mapping.json", 'r') as f:
            feature_mapping = json.load(f)
        print("✅ Feature mapping loaded")
    
    return data, metadata, feature_mapping

def prepare_features(data, feature_mapping):
    """Prepare features using static mapping"""
    print("🔧 Preparing features...")
    
    # Drop transaction_id if present
    feature_data = data.drop(['transaction_id'], axis=1, errors='ignore')
    
    # Separate features and target
    X = feature_data.drop("is_fraud", axis=1)
    y = feature_data["is_fraud"]
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=['merchant', 'location', 'card_type'])
    
    # Ensure consistent feature columns using mapping
    if feature_mapping and 'feature_columns' in feature_mapping:
        expected_columns = feature_mapping['feature_columns']
        
        # Add missing columns
        for col in expected_columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Reorder columns
        X_encoded = X_encoded.reindex(columns=expected_columns, fill_value=0)
        print(f"✅ Features aligned with mapping ({len(expected_columns)} features)")
    
    print(f"📈 Final features: {len(X_encoded.columns)}")
    print(f"📊 Fraud rate: {y.mean():.2%}")
    
    return X_encoded, y

def train_model(model, model_name, X_train, X_test, y_train, y_test, use_scaling=False):
    """Train and evaluate a single model"""
    
    print(f"\n📚 Training {model_name}...")
    
    with mlflow.start_run(run_name=f"{model_name}_{DOCKER_IMAGE_TAG}"):
        
        # Scale features if needed
        if use_scaling:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
            
            # Save scaler
            os.makedirs('models', exist_ok=True)
            scaler_path = f'models/{model_name}_scaler.pkl'
            joblib.dump(scaler, scaler_path)
        else:
            X_train_processed = X_train
            X_test_processed = X_test
            scaler_path = None
        
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("docker_tag", DOCKER_IMAGE_TAG)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("scaling_used", use_scaling)
        
        # Train model
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # AUC if probability available
        auc = 0.0
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        if auc > 0:
            mlflow.log_metric("roc_auc", auc)
        
        # Log model
        sample_input = X_test.iloc[:5] if hasattr(X_test, 'iloc') else X_test[:5]
        signature = mlflow.models.infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            model, 
            f"{model_name}_model",
            signature=signature,
            input_example=sample_input
        )
        
        auc_str = f"{auc:.4f}" if auc > 0 else "N/A"
        print(f"✅ {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_str}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'run_id': mlflow.active_run().info.run_id,
            'model_name': model_name,
            'scaler': scaler_path
        }

def train_models():
    """Main training function using static data"""
    print(f"🚀 Training models with static data - Experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    # Setup MLflow
    setup_mlflow()
    
    # Load static data
    data, metadata, feature_mapping = load_static_data()
    
    # Prepare features
    X, y = prepare_features(data, feature_mapping)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    
    # Define models
    models = {
        "random_forest": {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'use_scaling': False
        },
        "logistic_regression": {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'use_scaling': True
        },
        "gradient_boosting": {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'use_scaling': False
        }
    }
    
    print(f"\n🔄 Training {len(models)} models...")
    
    results = []
    
    # Train all models
    for name, config in models.items():
        try:
            result = train_model(
                config['model'], name, X_train, X_test, y_train, y_test, config['use_scaling']
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue
    
    if not results:
        raise Exception("❌ No models trained successfully")
    
    # Find best model
    best_result = max(results, key=lambda x: x['f1_score'])
    
    print(f"\n🏆 TRAINING COMPLETE")
    print(f"✅ Best model: {best_result['model_name']}")
    print(f"✅ Best F1 score: {best_result['f1_score']:.4f}")
    print(f"✅ Best accuracy: {best_result['accuracy']:.4f}")
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_result['model'], "models/best_model.pkl")
    print(f"✅ Model saved: models/best_model.pkl")
    
    # Save metadata
    model_metadata = {
        'best_model': best_result['model_name'],
        'accuracy': float(best_result['accuracy']),
        'f1_score': float(best_result['f1_score']),
        'auc': float(best_result['auc']),
        'run_id': best_result['run_id'],
        'docker_tag': DOCKER_IMAGE_TAG,
        'features': list(X.columns),
        'n_features': len(X.columns),
        'timestamp': datetime.now().isoformat(),
        'data_source': 'static_dataset',
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print("✅ Model metadata saved")
    return best_result

if __name__ == "__main__":
    try:
        best_model = train_models()
        print(f"\n🎉 Training successful! Best model: {best_model['model_name']}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        exit(1)
