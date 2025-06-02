output "vpc_id" {
  value = module.vpc.vpc_id
  description = "The ID of the created VPC"
}

output "public_subnets" {
  value = module.vpc.public_subnets
  description = "List of public subnets"
}

output "eks_cluster_name" {
  value = aws_eks_cluster.eks_cluster.name
  description = "EKS cluster name"
}

output "eks_cluster_endpoint" {
  value = aws_eks_cluster.eks_cluster.endpoint
  description = "EKS cluster endpoint (API server URL)"
}

output "eks_cluster_certificate_authority_data" {
  value = aws_eks_cluster.eks_cluster.certificate_authority[0].data
  description = "Base64 encoded certificate data required to access the cluster"
}

output "ecr_repo_url" {
  value = aws_ecr_repository.fraud_detection_repo.repository_url
  description = "ECR repository URL for pushing docker images"
}
