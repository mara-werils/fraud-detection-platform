variable "region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "Name of the EKS cluster and resource prefix"
  type        = string
  default     = "fraud-detection"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "instance_types" {
  description = "EC2 instance types for EKS node groups"
  type        = list(string)
  default     = ["m5.xlarge"]
}

variable "min_nodes" {
  description = "Minimum number of nodes in the EKS node group"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of nodes in the EKS node group"
  type        = number
  default     = 10
}

variable "desired_nodes" {
  description = "Desired number of nodes in the EKS node group"
  type        = number
  default     = 3
}

variable "clickhouse_instance_type" {
  description = "EC2 instance type for ClickHouse"
  type        = string
  default     = "r5.xlarge"
}
