# Use the official Python 3.8 image
FROM python:3.8-slim

# Install system dependencies (if needed for certain libraries like scikit-learn)
RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY docker/requirements.txt /app/requirements.txt

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of your application files (from the root project folder)
COPY . /app

# Expose both BentoML port (3000) and MLflow UI port (5000)
EXPOSE 3000 5000

# Run BentoML API server and MLflow UI
CMD ["sh", "-c", "bentoml serve fraud_detector:latest --port 3000 & mlflow ui --host 0.0.0.0 --port 5000"]
