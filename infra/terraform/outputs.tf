output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.k8s_cluster.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.k8s_cluster.cluster_name
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.k8s_cluster.cluster_security_group_id
}

output "kafka_bootstrap_brokers" {
  description = "Kafka (MSK) bootstrap broker connection string"
  value       = module.kafka.bootstrap_brokers
}

output "redis_endpoint" {
  description = "ElastiCache Redis primary endpoint"
  value       = module.redis.primary_endpoint
}

output "s3_data_lake_bucket" {
  description = "S3 bucket name for the data lake"
  value       = module.s3.bucket_name
}

output "clickhouse_endpoint" {
  description = "ClickHouse instance endpoint"
  value       = module.clickhouse.endpoint
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}
