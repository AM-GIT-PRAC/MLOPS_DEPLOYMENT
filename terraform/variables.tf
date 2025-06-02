##############################
# General Configuration
##############################
variable "region" {
  type        = string
  description = "AWS region"
}

variable "aws_account_id" {
  type        = string
  description = "Your AWS account ID"
}

##############################
# VPC Configuration
##############################
variable "vpc_name" {
  type        = string
  description = "Name of the VPC"
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC"
}

variable "availability_zones" {
  type        = list(string)
  description = "List of availability zones"
}

variable "public_subnets" {
  type        = list(string)
  description = "List of public subnet CIDRs"
}

##############################
# EKS Cluster Configuration
##############################
variable "cluster_name" {
  type        = string
  description = "Name of the EKS cluster"
}

##############################
# Node Group Configuration
##############################
variable "node_group_name" {
  type        = string
  description = "Name of the EKS node group"
}

variable "node_instance_type" {
  type        = string
  description = "EC2 instance type for nodes"
}

variable "desired_capacity" {
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

##############################
# ECR Configuration
##############################
variable "ecr_repo_name" {
  type        = string
  description = "Name of the ECR repository"
}
