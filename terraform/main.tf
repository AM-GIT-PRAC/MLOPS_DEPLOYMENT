provider "aws" {
  region = var.aws_region
}

##############################
# VPC
##############################
resource "aws_vpc" "main_vpc" {
  cidr_block           = var.vpc_cidr_block
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = {
    Name = "mlops-vpc"
  }
}

resource "aws_subnet" "public_subnets" {
  count             = length(var.public_subnet_cidrs)
  vpc_id            = aws_vpc.main_vpc.id
  cidr_block        = var.public_subnet_cidrs[count.index]
  availability_zone = element(data.aws_availability_zones.available.names, count.index)

  tags = {
    Name = "mlops-public-subnet-${count.index}"
  }
}

data "aws_availability_zones" "available" {}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main_vpc.id
  tags = {
    Name = "mlops-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main_vpc.id
  tags = {
    Name = "mlops-public-rt"
  }
}

resource "aws_route" "internet_access" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.gw.id
}

resource "aws_route_table_association" "public_subnet_assoc" {
  count          = length(var.public_subnet_cidrs)
  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public.id
}

##############################
# ECR
##############################
resource "aws_ecr_repository" "mlops_repo" {
  name = var.ecr_repo_name

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "mlops-ecr"
  }
}

##############################
# EKS Cluster
##############################
resource "aws_eks_cluster" "mlops_cluster" {
  name     = var.eks_cluster_name
  role_arn = aws_iam_role.eks_cluster_role.arn

  vpc_config {
    subnet_ids = aws_subnet.public_subnets[*].id
  }

  depends_on = [aws_iam_role_policy_attachment.eks_cluster_AmazonEKSClusterPolicy]
}

resource "aws_eks_node_group" "mlops_nodes" {
  cluster_name    = aws_eks_cluster.mlops_cluster.name
  node_group_name = var.eks_node_group_name
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = aws_subnet.public_subnets[*].id
  instance_types  = var.instance_types
  scaling_config {
    desired_size = 2
    max_size     = 2
    min_size     = 1
  }

  depends_on = [
    aws_eks_cluster.mlops_cluster,
    aws_iam_role_policy_attachment.eks_worker_node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.eks_worker_node_AmazonEC2ContainerRegistryReadOnly,
    aws_iam_role_policy_attachment.eks_worker_node_AmazonEKS_CNI_Policy
  ]
}

##############################
# IAM for EKS Cluster
##############################
resource "aws_iam_role" "eks_cluster_role" {
  name = "eksClusterRole"

  assume_role_policy = data.aws_iam_policy_document.eks_assume_role_policy.json
}

data "aws_iam_policy_document" "eks_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "eks_cluster_AmazonEKSClusterPolicy" {
  role       = aws_iam_role.eks_cluster_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

##############################
# IAM for EKS Node Group
##############################
resource "aws_iam_role" "eks_node_role" {
  name = "eksNodeGroupRole"

  assume_role_policy = data.aws_iam_policy_document.eks_node_assume_role.json
}

data "aws_iam_policy_document" "eks_node_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_AmazonEKSWorkerNodePolicy" {
  role       = aws_iam_role.eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_AmazonEC2ContainerRegistryReadOnly" {
  role       = aws_iam_role.eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_AmazonEKS_CNI_Policy" {
  role       = aws_iam_role.eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}
