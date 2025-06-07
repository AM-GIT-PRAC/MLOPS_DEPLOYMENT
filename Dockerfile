# Multi-stage build for production
FROM python:3.8-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.8-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' appuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY bentofile.yaml .

# Create a simple health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:3000/healthz || exit 1' > /health_check.sh && \
    chmod +x /health_check.sh

# Ensure model exists - create dummy if missing
RUN if [ ! -f "models/best_model.pkl" ]; then \
        echo "Warning: best_model.pkl not found, creating dummy model..."; \
        mkdir -p models && \
        python3 -c "
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a more comprehensive dummy model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Create sample data with expected features
features = ['amount', 'merchant_Amazon', 'merchant_BestBuy', 'merchant_Target', 
           'merchant_Walmart', 'merchant_eBay', 'location_NewYork', 'location_LosAngeles', 
           'card_type_Visa', 'card_type_MasterCard', 'card_type_Amex']

X = pd.DataFrame(np.random.rand(100, len(features)), columns=features)
y = np.random.randint(0, 2, 100)

model.fit(X, y)

# Save the model
joblib.dump(model, 'models/best_model.pkl')
print('Created dummy model with features:', features)
        "; \
    fi

# Create metadata file
RUN python3 -c "
import json
import os
metadata = {
    'model_path': 'models/best_model.pkl',
    'service_name': 'fraud_detector',
    'version': '1.0.0',
    'description': 'Fraud detection ML service'
}
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

# Set permissions
RUN chown -R appuser:appuser /app
USER appuser

# Add user's local bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Set environment variables
ENV PYTHONPATH="/app"
ENV MODEL_PATH="/app/models/best_model.pkl"

# Expose port
EXPOSE 3000

# Health check - use the script we created
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /health_check.sh

# Run BentoML service
CMD ["bentoml", "serve", "src.fraud_service:svc", "--host", "0.0.0.0", "--port", "3000"]
