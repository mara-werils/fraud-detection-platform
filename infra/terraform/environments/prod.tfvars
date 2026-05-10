region                   = "us-east-1"
cluster_name             = "fraud-detection-prod"
environment              = "prod"
instance_types           = ["m5.xlarge", "m5.2xlarge"]
min_nodes                = 3
max_nodes                = 20
desired_nodes            = 5
clickhouse_instance_type = "r5.2xlarge"
