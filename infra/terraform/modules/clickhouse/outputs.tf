output "endpoint" {
  description = "ClickHouse instance private IP"
  value       = aws_instance.clickhouse.private_ip
}

output "http_port" {
  description = "ClickHouse HTTP port"
  value       = 8123
}

output "native_port" {
  description = "ClickHouse native TCP port"
  value       = 9000
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.clickhouse.id
}
