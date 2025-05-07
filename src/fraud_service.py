import bentoml
from bentoml.io import JSON
import pandas as pd
import joblib

# Load model
model = joblib.load("models/best_model.pkl")

# Create service object
svc = bentoml.Service("fraud_detector")

# Define input/output format using updated style
@svc.api(input=JSON(), output=JSON())
def predict(input_data: dict) -> dict:
    df = pd.DataFrame([input_data])
    df_encoded = pd.get_dummies(df)

    # Align columns with training data
    required_columns = model.feature_names_in_
    for col in required_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[required_columns]

    prediction = model.predict(df_encoded)[0]
    return {"prediction": prediction}
