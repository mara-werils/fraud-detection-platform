region                   = "us-east-1"
cluster_name             = "fraud-detection-dev"
environment              = "dev"
instance_types           = ["m5.large"]
min_nodes                = 1
max_nodes                = 5
desired_nodes            = 2
clickhouse_instance_type = "r5.large"
