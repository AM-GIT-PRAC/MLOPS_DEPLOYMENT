# terraform.tfvars

#######################
# General AWS Config
#######################
aws_region      = "us-east-2"
aws_account_id  = "354918408969"

#######################
# VPC Configuration
#######################
vpc_cidr_block        = "10.0.0.0/16"
public_subnet_cidrs   = ["10.0.1.0/24", "10.0.2.0/24"]

#######################
# EKS Cluster
#######################
eks_cluster_name      = "mlops-eks-cluster"
eks_node_group_name   = "mlops-node-group"
instance_types        = ["t3.medium"]

#######################
# ECR Repository
#######################
ecr_repo_name         = "fraud-detection"
