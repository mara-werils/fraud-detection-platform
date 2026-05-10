data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "clickhouse" {
  name_prefix = "${var.cluster_name}-clickhouse-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 8123
    to_port     = 8123
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "ClickHouse HTTP"
  }

  ingress {
    from_port   = 9000
    to_port     = 9000
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "ClickHouse native TCP"
  }

  ingress {
    from_port   = 9009
    to_port     = 9009
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "ClickHouse inter-server replication"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.cluster_name}-clickhouse-sg"
    Environment = var.environment
  }
}

resource "aws_iam_role" "clickhouse" {
  name = "${var.cluster_name}-clickhouse-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "clickhouse_ssm" {
  role       = aws_iam_role.clickhouse.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "clickhouse" {
  name = "${var.cluster_name}-clickhouse-profile"
  role = aws_iam_role.clickhouse.name
}

resource "aws_ebs_volume" "clickhouse_data" {
  availability_zone = "${data.aws_region.current.name}a"
  size              = 500
  type              = "gp3"
  iops              = 3000
  throughput        = 250
  encrypted         = true

  tags = {
    Name        = "${var.cluster_name}-clickhouse-data"
    Environment = var.environment
  }
}

data "aws_region" "current" {}

resource "aws_instance" "clickhouse" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_ids[0]
  vpc_security_group_ids = [aws_security_group.clickhouse.id]
  iam_instance_profile   = aws_iam_instance_profile.clickhouse.name

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e

    # Install ClickHouse
    apt-get update
    apt-get install -y apt-transport-https ca-certificates dirmngr
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 8919F6BD2B48D754
    echo "deb https://packages.clickhouse.com/deb stable main" > /etc/apt/sources.list.d/clickhouse.list
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y clickhouse-server clickhouse-client

    # Mount data volume
    mkfs.ext4 /dev/xvdf || true
    mkdir -p /var/lib/clickhouse
    mount /dev/xvdf /var/lib/clickhouse
    echo '/dev/xvdf /var/lib/clickhouse ext4 defaults,nofail 0 2' >> /etc/fstab
    chown -R clickhouse:clickhouse /var/lib/clickhouse

    # Configure ClickHouse to listen on all interfaces
    sed -i 's/<!-- <listen_host>0.0.0.0<\/listen_host> -->/<listen_host>0.0.0.0<\/listen_host>/' /etc/clickhouse-server/config.xml

    # Start ClickHouse
    systemctl enable clickhouse-server
    systemctl start clickhouse-server
  EOF

  tags = {
    Name        = "${var.cluster_name}-clickhouse"
    Environment = var.environment
  }
}

resource "aws_volume_attachment" "clickhouse_data" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.clickhouse_data.id
  instance_id = aws_instance.clickhouse.id
}
