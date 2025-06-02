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

# Ensure model exists
RUN if [ ! -f "models/best_model.pkl" ]; then \
        echo "Warning: best_model.pkl not found, creating dummy model..."; \
        mkdir -p models && \
        python3 -c "import joblib; from sklearn.ensemble import RandomForestClassifier; import pandas as pd; \
        model = RandomForestClassifier(); \
        X = pd.DataFrame({'amount': [100], 'merchant_Amazon': [1], 'merchant_Walmart': [0], 'location_NewYork': [1], 'card_type_Visa': [1]}); \
        y = [0]; model.fit(X, y); \
        joblib.dump(model, 'models/best_model.pkl')"; \
    fi

# Set permissions
RUN chown -R appuser:appuser /app
USER appuser

# Add user's local bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/healthz || exit 1

# Run BentoML service
CMD ["bentoml", "serve", "src.fraud_service:svc", "--host", "0.0.0.0", "--port", "3000"]
