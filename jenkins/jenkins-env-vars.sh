#!/bin/bash

# === AWS CONFIGURATION ===
export AWS_REGION="us-east-2"
export AWS_ACCOUNT_ID="354918408969"
export ECR_REPO_NAME="fraud-detection"
export ECR_REPO_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

# === GITHUB CONFIGURATION ===
export GITHUB_REPO_URL="https://github.com/AM-GIT-PRAC/MLOPS.git"
export GIT_BRANCH="main"

# === SONARQUBE CONFIGURATION ===
export SONAR_PROJECT_KEY="mlops-legoland"
export SONAR_HOST_URL="http://18.224.61.192:9000"

# === KUBERNETES CONFIGURATION ===
export K8S_DEPLOYMENT_PATH="k8s/deployment.yaml"
export K8S_SERVICE_PATH="k8s/service.yaml"
export K8S_NAMESPACE="default"

# === DOCKER CONFIGURATION ===
export DOCKER_IMAGE_TAG="latest"

# === TERRAFORM CONFIGURATION ===
export TERRAFORM_DIR="terraform"

# === ML SERVICE CONFIGURATION ===
export MODEL_PORT="3000"
export MLFLOW_PORT="5000"
