##############################
# General Configuration
##############################
region         = "us-east-2"
aws_account_id = "354918408969" # ✅ Replace with your actual AWS account ID

##############################
# VPC Configuration
##############################
vpc_name           = "eks-vpc"
vpc_cidr           = "10.0.0.0/16"
availability_zones = ["us-east-2a", "us-east-2b"]
public_subnets     = ["10.0.1.0/24", "10.0.2.0/24"]

##############################
# EKS Cluster Configuration
##############################
cluster_name = "fraud-detection-cluster"

##############################
# Node Group Configuration
##############################
node_group_name    = "eks-node-group"
node_instance_type = "t3.medium"
desired_capacity   = 1
min_size           = 1
max_size           = 1

##############################
# ECR Configuration
##############################
ecr_repo_name = "fraud_detection"
