# BrahminyKite Monitoring & Metrics

Comprehensive monitoring and metrics collection for the BrahminyKite multi-framework AI verification system.

## Overview

The monitoring system provides:
- **Prometheus metrics** for all services and components
- **Custom metrics** for AI verification performance
- **Grafana dashboards** for visualization
- **Alerting rules** for operational monitoring
- **Health checks** and service discovery
- **Performance profiling** and troubleshooting

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Services      │    │   Consensus     │    │   API Gateway   │
│                 │    │   Network       │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  Metrics    │ │    │ │  Metrics    │ │    │ │  Metrics    │ │
│ │  Exporter   │ │    │ │  Exporter   │ │    │ │  Middleware │ │
│ └──────┬──────┘ │    │ └──────┬──────┘ │    │ └──────┬──────┘ │
└────────┼────────┘    └────────┼────────┘    └────────┼────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     Prometheus        │
                    │    (Metrics Store)    │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Grafana          │
                    │   (Visualization)     │
                    └───────────────────────┘
```

## Metrics Categories

### 1. **System Metrics**
- CPU, memory, disk usage
- Network I/O and latency
- Process-level resource consumption
- Container resource limits

### 2. **Service Metrics**
- Request/response rates and latency
- Error rates and status codes
- Service availability and health
- Queue depths and processing times

### 3. **gRPC Metrics**
- RPC call rates and latency
- Connection pool statistics
- Streaming connection metrics
- Error rates by method

### 4. **AI Framework Metrics**
- Verification accuracy and confidence
- Model inference latency
- Training metrics and convergence
- Evidence processing times

### 5. **Consensus Metrics**
- Proposal rates and success ratios
- Leader election frequency
- Network partition detection
- State synchronization times

### 6. **Business Metrics**
- Claims processed per framework
- Consensus decision accuracy
- User verification requests
- System throughput

## Usage

### Basic Metrics Collection

```python
from chil.monitoring import MetricsRegistry, Counter, Histogram, Gauge

# Create metrics registry
metrics = MetricsRegistry()

# Define custom metrics
verification_requests = Counter(
    'verification_requests_total',
    'Total verification requests',
    ['framework', 'status']
)

verification_latency = Histogram(
    'verification_latency_seconds',
    'Time spent processing verifications',
    ['framework']
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections',
    ['service']
)

# Record metrics
verification_requests.labels(framework='empirical', status='success').inc()
verification_latency.labels(framework='empirical').observe(0.25)
active_connections.labels(service='api').set(42)
```

### Service Integration

```python
from chil.monitoring import MetricsExporter

class EmpiricalService:
    def __init__(self):
        self.metrics_exporter = MetricsExporter(
            service_name='empirical',
            port=9090
        )
        self.setup_metrics()
    
    def setup_metrics(self):
        self.request_duration = Histogram(
            'service_request_duration_seconds',
            'Service request duration',
            ['method', 'status']
        )
    
    async def verify_claim(self, claim):
        with self.request_duration.labels(
            method='verify_claim', 
            status='success'
        ).time():
            # Process verification
            result = await self.process_verification(claim)
            return result
```

### FastAPI Integration

```python
from chil.monitoring import FastAPIMetricsMiddleware

app = FastAPI()

# Add metrics middleware
app.add_middleware(FastAPIMetricsMiddleware)

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

## Configuration

### Environment Variables

```bash
# Metrics Configuration
METRICS_ENABLED=true
METRICS_PORT=9090
METRICS_PATH=/metrics
METRICS_NAMESPACE=brahminykite

# Prometheus Configuration
PROMETHEUS_URL=http://prometheus:9090
PROMETHEUS_SCRAPE_INTERVAL=15s

# Grafana Configuration
GRAFANA_URL=http://grafana:3000
GRAFANA_API_KEY=your_api_key
```

### YAML Configuration

```yaml
monitoring:
  metrics:
    enabled: true
    port: 9090
    path: /metrics
    namespace: brahminykite
    
  exporters:
    - name: prometheus
      port: 9090
      path: /metrics
    
  collectors:
    - system_metrics
    - grpc_metrics
    - custom_metrics
    
  dashboards:
    auto_import: true
    directory: /etc/grafana/dashboards
    
  alerts:
    enabled: true
    rules_file: /etc/prometheus/alerts.yml
```

## Available Metrics

### Framework Services

```prometheus
# Request metrics
brahminykite_requests_total{service="empirical", method="verify_claim", status="success"}
brahminykite_request_duration_seconds{service="empirical", method="verify_claim"}

# Processing metrics
brahminykite_claims_processed_total{framework="empirical", result="valid"}
brahminykite_evidence_processing_seconds{framework="empirical", type="scientific"}

# Model metrics
brahminykite_model_inference_seconds{framework="empirical", model="mini_llm"}
brahminykite_model_confidence{framework="empirical", claim_id="123"}
```

### Consensus Network

```prometheus
# Network metrics
brahminykite_consensus_messages_sent_total{node_id="node-1", peer="node-2", type="prepare"}
brahminykite_consensus_network_latency_seconds{node_id="node-1", peer="node-2"}

# Consensus metrics
brahminykite_consensus_proposals_total{node_id="node-1", status="accepted"}
brahminykite_consensus_leader_elections_total{node_id="node-1"}
brahminykite_consensus_state_sync_seconds{node_id="node-1"}
```

### API Gateway

```prometheus
# HTTP metrics
brahminykite_http_requests_total{method="POST", endpoint="/verify", status="200"}
brahminykite_http_request_duration_seconds{method="POST", endpoint="/verify"}

# Rate limiting
brahminykite_rate_limit_hits_total{endpoint="/verify", limit_type="per_minute"}
brahminykite_rate_limit_remaining{endpoint="/verify", client_id="user123"}
```

## Grafana Dashboards

### 1. **System Overview**
- Service health and availability
- Resource usage across all components
- Error rates and response times
- Active connections and throughput

### 2. **AI Framework Performance**
- Verification accuracy by framework
- Processing latency distributions
- Model inference performance
- Evidence analysis metrics

### 3. **Consensus Network**
- Network topology and health
- Message flow and latency
- Leader election frequency
- State synchronization status

### 4. **API Performance**
- Request rates and patterns
- Response time percentiles
- Error rates by endpoint
- Rate limiting effectiveness

## Alerting Rules

### Critical Alerts

```yaml
- alert: ServiceDown
  expr: up{job="brahminykite-services"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "BrahminyKite service is down"

- alert: HighErrorRate
  expr: rate(brahminykite_requests_total{status!="success"}[5m]) > 0.1
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High error rate detected"

- alert: ConsensusPartition
  expr: brahminykite_consensus_active_peers < 2
  for: 30s
  labels:
    severity: critical
  annotations:
    summary: "Consensus network partition detected"
```

## Development

### Running Locally

```bash
# Start monitoring stack
docker-compose up -d prometheus grafana

# Access dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus

# Import dashboards
./scripts/import-dashboards.sh
```

### Custom Metrics

```python
from chil.monitoring import register_metric

@register_metric
class CustomVerificationMetric:
    def __init__(self):
        self.accuracy_gauge = Gauge(
            'verification_accuracy',
            'Current verification accuracy',
            ['framework', 'claim_type']
        )
    
    def update_accuracy(self, framework, claim_type, accuracy):
        self.accuracy_gauge.labels(
            framework=framework,
            claim_type=claim_type
        ).set(accuracy)
```

## Best Practices

1. **Metric Naming**: Use consistent naming with service prefix
2. **Labels**: Keep cardinality low, avoid high-cardinality labels
3. **Sampling**: Use histograms for latency, counters for events
4. **Retention**: Configure appropriate retention for your use case
5. **Alerting**: Alert on symptoms, not causes
6. **Documentation**: Document custom metrics and their purpose