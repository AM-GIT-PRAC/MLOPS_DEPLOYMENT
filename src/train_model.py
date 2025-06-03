import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime

# Load configuration from environment or use defaults
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'fraud-detection-production')
MODEL_NAME = os.getenv('MODEL_NAME', 'fraud_detector')
DOCKER_IMAGE_TAG = os.getenv('DOCKER_IMAGE_TAG', f'v{datetime.now().strftime("%Y%m%d-%H%M%S")}')

# Set MLflow tracking directory
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

def train_models():
    print(f"🚀 Starting model training for experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    # Load data
    if not os.path.exists("data/transactions.csv"):
        print("❌ Data file not found. Run generate_data.py first.")
        return
    
    data = pd.read_csv("data/transactions.csv")
    print(f"📊 Loaded {len(data)} transactions")
    
    # Prepare features
    X = pd.get_dummies(data.drop("is_fraud", axis=1))
    y = data["is_fraud"].map({'FRA': 1, 'NOR': 0})
    
    print(f"📈 Features: {list(X.columns)}")
    print(f"📊 Fraud rate: {y.mean():.2%}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Models to train
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Start MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_scores = {}
    model_objects = {}

    print(f"\n🔄 Training {len(models)} models...")
    
    for name, model in models.items():
        print(f"\n📚 Training {name}...")
        
        with mlflow.start_run(run_name=f"{name}-{DOCKER_IMAGE_TAG}"):
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("model_name", name)
            mlflow.log_param("docker_tag", DOCKER_IMAGE_TAG)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_metric("accuracy", acc)
            
            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_scores[mlflow.active_run().info.run_id] = acc
            model_objects[mlflow.active_run().info.run_id] = model

            print(f"✅ {name} - Accuracy: {acc:.4f}")

    # Select best model
    best_run_id = max(run_scores, key=run_scores.get)
    best_accuracy = run_scores[best_run_id]
    best_model = model_objects[best_run_id]

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    print(f"\n🏆 TRAINING COMPLETE")
    print(f"✅ Best model run ID: {best_run_id}")
    print(f"✅ Best accuracy: {best_accuracy:.4f}")
    print(f"✅ Model saved to: models/best_model.pkl")
    print(f"✅ Docker tag: {DOCKER_IMAGE_TAG}")
    
    return best_model, best_accuracy

if __name__ == "__main__":
    train_models()
