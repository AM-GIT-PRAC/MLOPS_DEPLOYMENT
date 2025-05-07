# src/train_model.py

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#  Set MLflow logging directory
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

#  Load data
data = pd.read_csv("data/transactions.csv")
X = pd.get_dummies(data.drop("is_fraud", axis=1))
y = data["is_fraud"]

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Models to train
models = {
    "random_forest": RandomForestClassifier(),
    "logistic_regression": LogisticRegression(max_iter=200),
    "gradient_boosting": GradientBoostingClassifier()
}

#  Start MLflow experiment
mlflow.set_experiment("fraud-detection-legoland")
run_scores = {}
model_objects = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_scores[mlflow.active_run().info.run_id] = acc
        model_objects[mlflow.active_run().info.run_id] = model

        print(f"{name} trained. Accuracy: {acc}")

#  Select best model
best_run_id = max(run_scores, key=run_scores.get)
best_accuracy = run_scores[best_run_id]
best_model = model_objects[best_run_id]

#  Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

print(f"\n Best model run ID: {best_run_id} with Accuracy: {best_accuracy}")
print(" Best model saved to models/best_model.pkl")
