#######################
# General AWS Config
#######################
variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "aws_account_id" {
  description = "Your AWS account ID"
  type        = string
}

#######################
# VPC Configuration
#######################
variable "vpc_cidr_block" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "public_subnet_cidrs" {
  description = "List of public subnet CIDRs"
  type        = list(string)
}

#######################
# EKS Cluster
#######################
variable "eks_cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "eks_node_group_name" {
  description = "Name for the EKS node group"
  type        = string
}

variable "instance_types" {
  description = "EC2 instance types for EKS node group"
  type        = list(string)
}

#######################
# ECR Repository
#######################
variable "ecr_repo_name" {
  description = "ECR repository name"
  type        = string
}
