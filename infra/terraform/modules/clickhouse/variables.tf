variable "cluster_name" {
  description = "Name prefix for ClickHouse resources"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs"
  type        = list(string)
}

variable "instance_type" {
  description = "EC2 instance type for ClickHouse"
  type        = string
  default     = "r5.xlarge"
}
