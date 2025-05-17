#############################
#       GENERAL CONFIG     #
#############################

variable "region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-2"
}

variable "aws_account_id" {
  description = "AWS Account ID used for ECR"
  type        = string
}


#############################
#         VPC CONFIG       #
#############################

variable "vpc_name" {
  description = "Name of the VPC"
  type        = string
  default     = "eks-vpc"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "azs" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-2a", "us-east-2b"]
}

variable "public_subnets" {
  description = "Public subnet CIDRs"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}


#############################
#      EKS CONFIG           #
#############################

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "fraud-detection-eks"
}

variable "node_group_name" {
  description = "Name of the EKS node group"
  type        = string
  default     = "eks-node-group"
}

variable "instance_type" {
  description = "EC2 instance type for EKS nodes"
  type        = string
  default     = "t3.medium"
}

variable "min_size" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}

variable "max_size" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 1
}

variable "desired_size" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 1
}


#############################
#        ECR CONFIG         #
#############################

variable "ecr_repo_name" {
  description = "Name of the ECR repository"
  type        = string
  default     = "fraud-detection"
}
