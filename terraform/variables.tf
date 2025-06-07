# terraform/variables.tf
# Variable definitions only - no default values

variable "region" {
  type        = string
  description = "AWS region"
}

variable "aws_account_id" {
  type        = string
  description = "Your AWS account ID"
}

variable "vpc_name" {
  type        = string
  description = "Name of the VPC"
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC"
}

variable "azs" {
  type        = list(string)
  description = "List of availability zones"
}

variable "public_subnets" {
  type        = list(string)
  description = "List of public subnet CIDRs"
}

variable "private_subnets" {
  type        = list(string)
  description = "List of private subnet CIDRs"
}

variable "cluster_name" {
  type        = string
  description = "Name of the EKS cluster"
}

variable "node_group_name" {
  type        = string
  description = "Name of the EKS node group"
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type for nodes"
}

variable "desired_size" {
  type        = number
  description = "Desired number of nodes"
}

variable "min_size" {
  type        = number
  description = "Minimum number of nodes"
}

variable "max_size" {
  type        = number
  description = "Maximum number of nodes"
}

variable "key_name" {
  type        = string
  description = "EC2 Key Pair name for SSH access (optional)"
}

variable "ecr_repo_name" {
  type        = string
  description = "Name of the ECR repository"
}
