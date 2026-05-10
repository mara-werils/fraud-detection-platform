resource "helm_release" "prometheus_stack" {
  name             = "prometheus"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = "55.5.0"
  namespace        = "monitoring"
  create_namespace = true

  values = [
    file("${path.module}/../../k8s/monitoring/prometheus-values.yaml")
  ]

  set {
    name  = "grafana.adminPassword"
    value = var.grafana_admin_password
  }

  set {
    name  = "prometheus.prometheusSpec.retention"
    value = "30d"
  }

  set {
    name  = "prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage"
    value = "50Gi"
  }
}

resource "helm_release" "prometheus_adapter" {
  name       = "prometheus-adapter"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "prometheus-adapter"
  version    = "4.9.0"
  namespace  = "monitoring"

  set {
    name  = "prometheus.url"
    value = "http://prometheus-kube-prometheus-prometheus.monitoring.svc"
  }

  set {
    name  = "prometheus.port"
    value = "9090"
  }

  values = [
    <<-EOT
    rules:
      custom:
        - seriesQuery: 'scoring_request_duration_seconds_bucket{namespace!="",pod!=""}'
          resources:
            overrides:
              namespace: {resource: "namespace"}
              pod: {resource: "pod"}
          name:
            matches: "^(.*)_bucket$"
            as: "scoring_latency_p99"
          metricsQuery: 'histogram_quantile(0.99, sum(rate(<<.Series>>{<<.LabelMatchers>>}[5m])) by (le, <<.GroupBy>>))'
    EOT
  ]

  depends_on = [helm_release.prometheus_stack]
}

resource "helm_release" "jaeger" {
  name             = "jaeger"
  repository       = "https://jaegertracing.github.io/helm-charts"
  chart            = "jaeger"
  version          = "2.0.0"
  namespace        = "monitoring"
  create_namespace = false

  set {
    name  = "provisionDataStore.cassandra"
    value = "false"
  }

  set {
    name  = "allInOne.enabled"
    value = "true"
  }

  set {
    name  = "storage.type"
    value = "memory"
  }

  set {
    name  = "agent.enabled"
    value = "false"
  }

  set {
    name  = "collector.enabled"
    value = "false"
  }

  set {
    name  = "query.enabled"
    value = "false"
  }

  depends_on = [helm_release.prometheus_stack]
}

resource "kubernetes_config_map" "grafana_fraud_dashboards" {
  metadata {
    name      = "grafana-fraud-dashboards"
    namespace = "monitoring"
    labels = {
      grafana_dashboard = "1"
    }
  }

  data = {
    "fraud-overview.json"     = file("${path.module}/../../k8s/monitoring/grafana-dashboards/fraud-overview.json")
    "model-performance.json"  = file("${path.module}/../../k8s/monitoring/grafana-dashboards/model-performance.json")
    "system-health.json"      = file("${path.module}/../../k8s/monitoring/grafana-dashboards/system-health.json")
  }

  depends_on = [helm_release.prometheus_stack]
}
