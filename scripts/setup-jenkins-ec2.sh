#!/bin/bash

echo "🚀 Setting up Jenkins on EC2 with all required tools..."

# Exit on any error
set -e

# Update system
sudo apt update && sudo apt upgrade -y

# Install Java 17 (required for Jenkins)
echo "📦 Installing Java 17..."
sudo apt install -y openjdk-17-jdk

# Install Jenkins
echo "🔧 Installing Jenkins..."
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | sudo tee /usr/share/keyrings/jenkins-keyring.asc > /dev/null
echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/ | sudo tee /etc/apt/sources.list.d/jenkins.list > /dev/null
sudo apt update
sudo apt install -y jenkins

# Install Docker
echo "🐳 Installing Docker..."
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker jenkins
sudo usermod -aG docker $USER

# Install AWS CLI v2
echo "☁️ Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws

# Install kubectl
echo "⚓ Installing kubectl..."
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
echo "🎡 Installing Helm..."
curl -fsSL https://get.helm.sh/helm-v3.15.4-linux-amd64.tar.gz -o helm.tar.gz
tar -zxvf helm.tar.gz
sudo mv linux-amd64/helm /usr/local/bin/helm
rm -rf helm.tar.gz linux-amd64

# Install Terraform
echo "🏗️ Installing Terraform..."
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install -y terraform

# Install Python and ML dependencies
echo "🐍 Installing Python packages..."
sudo apt install -y python3 python3-pip python3-venv
pip3 install --user mlflow bentoml scikit-learn pandas numpy joblib faker

# Install Trivy for security scanning
echo "🔒 Installing Trivy..."
sudo apt-get install wget apt-transport-https gnupg lsb-release -y
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy -y

# Install SonarQube Scanner
echo "📊 Installing SonarQube Scanner..."
SONAR_SCANNER_VERSION=5.0.1.3006
wget -O /tmp/sonar-scanner.zip https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-${SONAR_SCANNER_VERSION}.zip
sudo unzip /tmp/sonar-scanner.zip -d /opt
sudo mv /opt/sonar-scanner-${SONAR_SCANNER_VERSION} /opt/sonar-scanner
sudo chmod +x /opt/sonar-scanner/bin/sonar-scanner
sudo ln -s /opt/sonar-scanner/bin/sonar-scanner /usr/local/bin/sonar-scanner
rm /tmp/sonar-scanner.zip

# Create Jenkins workspace directory with proper permissions
sudo mkdir -p /var/lib/jenkins/workspace
sudo chown -R jenkins:jenkins /var/lib/jenkins
sudo chmod -R 755 /var/lib/jenkins

# Add environment variables to Jenkins user profile
sudo -u jenkins bash -c 'echo "export PATH=\"\$HOME/.local/bin:/opt/sonar-scanner/bin:\$PATH\"" >> /var/lib/jenkins/.bashrc'

# Start Jenkins
echo "🚀 Starting Jenkins..."
sudo systemctl start jenkins
sudo systemctl enable jenkins

# Restart Docker to ensure group membership takes effect
sudo systemctl restart docker

# Wait for Jenkins to start
echo "⏳ Waiting for Jenkins to start..."
sleep 30

# Get Jenkins initial password
echo ""
echo "🎉 Setup Complete!"
echo "=================================="
echo "🔑 Jenkins initial admin password:"
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
echo ""
echo "🌐 Access Jenkins at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
echo ""
echo "📋 Next steps:"
echo "1. Access Jenkins web interface"
echo "2. Install suggested plugins"
echo "3. Create admin user"
echo "4. Install additional plugins: Docker Pipeline, Kubernetes, SonarQube Scanner"
echo "5. Configure credentials (GitHub, AWS, Kubeconfig, SonarQube)"
echo ""
echo "🔧 Tool versions installed:"
java -version
docker --version
aws --version
kubectl version --client
helm version
terraform version
trivy --version
sonar-scanner --version
python3 --version
echo ""
echo "⚠️ Remember to:"
echo "   - Log out and log back in for Docker group membership"
echo "   - Configure AWS credentials: aws configure"
echo "   - Set up kubeconfig for EKS access"
