import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix, roc_auc_score)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load configuration from environment or use defaults
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'fraud-detection-production')
MODEL_NAME = os.getenv('MODEL_NAME', 'fraud_detector')
DOCKER_IMAGE_TAG = os.getenv('DOCKER_IMAGE_TAG', f'v{datetime.now().strftime("%Y%m%d-%H%M%S")}')

def setup_mlflow():
    """Setup MLflow tracking"""
    # Try to connect to MLflow server, fallback to file-based tracking
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        # Test connection
        mlflow.get_experiment_by_name("test")
        print(f"✅ Connected to MLflow at: {mlflow_uri}")
    except Exception:
        # Fallback to file-based tracking
        mlflow.set_tracking_uri("file:./mlruns")
        print("⚠️ Using file-based MLflow tracking")
    
    # Set or create experiment
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
        print(f"⚠️ MLflow setup warning: {e}")
        return None

def load_and_prepare_data():
    """Load and prepare the training data"""
    print("📊 Loading and preparing data...")
    
    # Check if data exists
    if not os.path.exists("data/transactions.csv"):
        raise FileNotFoundError("❌ Data file not found. Run generate_data.py first.")
    
    # Load data
    data = pd.read_csv("data/transactions.csv")
    print(f"📈 Loaded {len(data)} transactions")
    
    # Load feature mapping if available
    feature_mapping = None
    if os.path.exists("data/feature_mapping.json"):
        with open("data/feature_mapping.json", 'r') as f:
            feature_mapping = json.load(f)
        print("✅ Feature mapping loaded")
    
    # Prepare features with one-hot encoding
    X = pd.get_dummies(data.drop("is_fraud", axis=1))
    y = data["is_fraud"]
    
    # Ensure consistent feature columns if mapping exists
    if feature_mapping and 'feature_columns' in feature_mapping:
        expected_columns = feature_mapping['feature_columns']
        # Add missing columns with zero values
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0
        # Reorder columns to match expected order
        X = X.reindex(columns=expected_columns, fill_value=0)
        print(f"✅ Features aligned with mapping ({len(expected_columns)} features)")
    
    print(f"📈 Features ({len(X.columns)}): {list(X.columns)[:10]}...")
    print(f"📊 Fraud rate: {y.mean():.2%}")
    
    return X, y, feature_mapping

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, 
                           feature_names, use_scaling=False):
    """Train and evaluate a single model with comprehensive metrics"""
    
    print(f"\n📚 Training {model_name}...")
    
    with mlflow.start_run(run_name=f"{model_name}_{DOCKER_IMAGE_TAG}") as run:
        
        # Scale features if needed (for logistic regression)
        if use_scaling:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
            # Save scaler
            joblib.dump(scaler, f'models/{model_name}_scaler.pkl')
            mlflow.log_artifact(f'models/{model_name}_scaler.pkl')
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Log parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            mlflow.log_params(params)
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("docker_tag", DOCKER_IMAGE_TAG)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features_count", len(feature_names))
        mlflow.log_param("scaling_used", use_scaling)
        
        # Train model
        model.fit(X_train_processed, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='accuracy')
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
            mlflow.log_metric("roc_auc", auc)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(class_report, f"{model_name}_classification_report.json")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_dict({
            "confusion_matrix": cm.tolist(),
            "labels": ["legitimate", "fraud"]
        }, f"{model_name}_confusion_matrix.json")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log top 20 features
            top_features = feature_importance.head(20).to_dict('records')
            mlflow.log_dict(top_features, f"{model_name}_feature_importance.json")
        
        # Create input example for model signature
        sample_input = X_test.iloc[:5] if hasattr(X_test, 'iloc') else X_test[:5]
        
        # Log model with signature
        signature = mlflow.models.infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            model, 
            f"{model_name}_model",
            signature=signature,
            input_example=sample_input
        )
        
        print(f"✅ {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f if y_pred_proba is not None else 'N/A'}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc if y_pred_proba is not None else 0,
            'run_id': run.info.run_id,
            'model_name': model_name,
            'scaler': f'models/{model_name}_scaler.pkl' if use_scaling else None
        }

def train_models():
    """Main training function"""
    print(f"🚀 Starting model training for experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    # Setup MLflow
    experiment_id = setup_mlflow()
    
    # Load and prepare data
    X, y, feature_mapping = load_and_prepare_data()
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Define models with optimized hyperparameters
    models = {
        "random_forest": {
            'model': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'use_scaling': False
        },
        "logistic_regression": {
            'model': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                solver='liblinear',
                class_weight='balanced'
            ),
            'use_scaling': True
        },
        "gradient_boosting": {
            'model': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'use_scaling': False
        }
    }

    print(f"\n🔄 Training {len(models)} models...")
    
    results = []
    
    # Train all models
    for name, config in models.items():
        try:
            result = train_and_evaluate_model(
                config['model'], 
                name, 
                X_train, X_test, y_train, y_test,
                list(X.columns),
                config['use_scaling']
            )
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue
    
    if not results:
        raise Exception("❌ No models were successfully trained")
    
    # Find best model (using F1 score as primary metric)
    best_result = max(results, key=lambda x: x['f1_score'])
    
    print(f"\n🏆 TRAINING COMPLETE")
    print(f"✅ Best model: {best_result['model_name']}")
    print(f"✅ Best F1 score: {best_result['f1_score']:.4f}")
    print(f"✅ Best accuracy: {best_result['accuracy']:.4f}")
    print(f"✅ Run ID: {best_result['run_id']}")
    
    # Save best model and metadata
    os.makedirs("models", exist_ok=True)
    
    # Save the best model
    joblib.dump(best_result['model'], "models/best_model.pkl")
    print(f"✅ Best model saved to: models/best_model.pkl")
    
    # Save model metadata
    metadata = {
        'best_model': best_result['model_name'],
        'accuracy': float(best_result['accuracy']),
        'f1_score': float(best_result['f1_score']),
        'auc': float(best_result['auc']),
        'run_id': best_result['run_id'],
        'docker_tag': DOCKER_IMAGE_TAG,
        'features': list(X.columns),
        'n_features': len(X.columns),
        'fraud_rate': float(y.mean()),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'timestamp': datetime.now().isoformat(),
        'scaler_path': best_result.get('scaler'),
        'mlflow_experiment': MLFLOW_EXPERIMENT_NAME
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Model metadata saved")
    
    # Print summary of all models
    print(f"\n📊 Model Performance Summary:")
    print("-" * 60)
    for result in sorted(results, key=lambda x: x['f1_score'], reverse=True):
        print(f"{result['model_name']:20} | F1: {result['f1_score']:.4f} | Acc: {result['accuracy']:.4f} | AUC: {result['auc']:.4f}")
    
    return best_result

if __name__ == "__main__":
    try:
        best_model_result = train_models()
        print(f"\n🎉 Training completed successfully!")
        print(f"Best model: {best_model_result['model_name']}")
        print(f"Docker tag: {DOCKER_IMAGE_TAG}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        exit(1)
