
# terraform/terraform.tfvars
# Corrected configuration with 2 AZs for EKS

# General Configuration
region         = "us-east-2"
aws_account_id = "354918408969"

# VPC Configuration - 2 AZs required for EKS
vpc_name        = "eks-vpc"
vpc_cidr        = "10.0.0.0/16"
azs             = ["us-east-2a", "us-east-2b"]  # 2 AZs for EKS
public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]  # 2 public subnets
private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]      # 2 private subnets

# EKS Cluster Configuration
cluster_name = "fraud-detection-cluster"

# Node Group Configuration - Single instance for testing
node_group_name = "eks-node-group"
instance_type   = "t3.small"
desired_size    = 1
min_size        = 1
max_size        = 2

# ECR Configuration
ecr_repo_name = "fraud-detection"
