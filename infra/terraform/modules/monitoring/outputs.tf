output "prometheus_endpoint" {
  description = "Prometheus server endpoint"
  value       = "http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090"
}

output "grafana_endpoint" {
  description = "Grafana endpoint"
  value       = "http://prometheus-grafana.monitoring.svc:80"
}

output "jaeger_endpoint" {
  description = "Jaeger query endpoint"
  value       = "http://jaeger-query.monitoring.svc:16686"
}
