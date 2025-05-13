
provider "aws" {
    region = var.region
}

#vpc, subnets, networking

module "vpc" {
    source = "terraform-aws-modules/vpc/aws"
    version = "5.19.0"
    map_public_ip_on_launch = true

    name = "eks-vpc"
    cidr = "10.0.0.0/16"

    azs = ["us-east-2a", "us-east-2b"]
    public_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

    enable_dns_hostnames = true
    enable_dns_support = true
}

#eks cluster
resource "aws_eks_cluster" "eks_cluster"{
    name        = var.cluster_name
    role_arn    = aws_iam_role.eks_cluster_role.arn
    vpc_config {
        subnet_ids = module.vpc.public_subnets
    }

    depends_on =[aws_iam_role_policy_attachment.eks_cluster_AmazonEKSClusterPolicy]
}

#NODE GROUPS
resource "aws_eks_node_group" "eks_nodes" {
    cluster_name = aws_eks_cluster.eks_cluster.name
    node_group_name = "eks-node-group"
    node_role_arn = aws_iam_role.eks_node_role.arn
    subnet_ids = module.vpc.public_subnets

    scaling_config {
        desired_size = 1
        max_size = 1
        min_size =1
    }

    instance_types =["t3.large"]
}


#Iam Roles/ Eks cluster roles

resource "aws_iam_role" "eks_cluster_role" {
    name = "eks_cluster_role"
    assume_role_policy = data.aws_iam_policy_document.eks_cluster_assume.json
}

resource "aws_iam_role_policy_attachment" "eks_cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster_role.name
}

resource "aws_iam_role" "eks_node_role" {
  name = "eks-node-role"
  assume_role_policy = data.aws_iam_policy_document.eks_nodes_assume.json
}

resource "aws_iam_role_policy_attachment" "eks_node_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_role.name
}

resource "aws_iam_role_policy_attachment" "eks_node_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_role.name
}

resource "aws_iam_role_policy_attachment" "eks_node_AmazonEKSCNIPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_role.name
}

# IAM Assume Role Policies
data "aws_iam_policy_document" "eks_cluster_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "eks_nodes_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}
