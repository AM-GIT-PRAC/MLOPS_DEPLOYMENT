output "vpc_id" {
  value = aws_vpc.main_vpc.id
}

output "ecr_repo_url" {
  value = aws_ecr_repository.mlops_repo.repository_url
}

output "eks_cluster_name" {
  value = aws_eks_cluster.mlops_cluster.name
}

output "eks_cluster_endpoint" {
  value = aws_eks_cluster.mlops_cluster.endpoint
}

output "eks_cluster_certificate_authority" {
  value = aws_eks_cluster.mlops_cluster.certificate_authority[0].data
}
