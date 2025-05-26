# BrahminyKite Helm Chart

This Helm chart deploys the BrahminyKite multi-framework AI verification system on a Kubernetes cluster.

## Prerequisites

- Kubernetes 1.23+
- Helm 3.10+
- PV provisioner support in the underlying infrastructure (for PostgreSQL and Prometheus persistence)
- cert-manager (optional, for automatic TLS certificate management)
- NGINX Ingress Controller (optional, for ingress support)

## Installation

### Add the Helm repository (when published)

```bash
helm repo add brahminykite https://charts.brahminykite.io
helm repo update
```

### Install from local chart

```bash
# Install with default values
helm install brahminykite ./helm/brahminykite

# Install with custom values
helm install brahminykite ./helm/brahminykite -f my-values.yaml

# Install in a specific namespace
helm install brahminykite ./helm/brahminykite --namespace brahminykite --create-namespace
```

### Install with different environments

```bash
# Development environment
helm install brahminykite-dev ./helm/brahminykite -f ./helm/brahminykite/values-dev.yaml

# Production environment
helm install brahminykite-prod ./helm/brahminykite -f ./helm/brahminykite/values-prod.yaml
```

## Configuration

The following table lists the configurable parameters and their default values.

### Global Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imagePullSecrets` | Global Docker registry secret names | `[]` |
| `global.storageClass` | Global storage class for persistent volumes | `""` |
| `global.nodeSelector` | Global node selector | `{}` |
| `global.tolerations` | Global tolerations | `[]` |
| `global.affinity` | Global affinity rules | `{}` |

### Framework Services

| Parameter | Description | Default |
|-----------|-------------|---------|
| `services.replicaCount` | Default replica count for all services | `2` |
| `services.image.registry` | Docker registry for service images | `docker.io` |
| `services.image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `services.image.tag` | Image tag | `1.0.0` |
| `services.resources` | Default resource requests/limits | See values.yaml |
| `services.<service>.enabled` | Enable specific service | `true` |
| `services.<service>.name` | Service name | varies |
| `services.<service>.port` | Service port | `50051-50056` |
| `services.<service>.replicaCount` | Service-specific replica count | `2` |

### API Gateway

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api.enabled` | Enable API Gateway | `true` |
| `api.replicaCount` | API replica count | `3` |
| `api.image.repository` | API image repository | `brahminykite/api` |
| `api.service.type` | Kubernetes service type | `ClusterIP` |
| `api.service.port` | API service port | `8000` |
| `api.resources` | API resource requests/limits | See values.yaml |

### Ingress

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `true` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.annotations` | Ingress annotations | See values.yaml |
| `ingress.hosts` | Ingress hosts configuration | See values.yaml |
| `ingress.tls` | TLS configuration | See values.yaml |

### Database (PostgreSQL)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `postgresql.enabled` | Deploy PostgreSQL | `true` |
| `postgresql.auth.username` | PostgreSQL username | `brahminykite` |
| `postgresql.auth.password` | PostgreSQL password | `brahminykite-prod-2024` |
| `postgresql.auth.database` | PostgreSQL database | `brahminykite` |
| `postgresql.primary.persistence.size` | PVC size | `10Gi` |

### Cache (Redis)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `redis.enabled` | Deploy Redis | `true` |
| `redis.auth.enabled` | Enable Redis authentication | `false` |
| `redis.master.persistence.enabled` | Enable Redis persistence | `false` |

### Monitoring

| Parameter | Description | Default |
|-----------|-------------|---------|
| `prometheus.enabled` | Deploy Prometheus | `true` |
| `grafana.enabled` | Deploy Grafana | `true` |
| `monitoring.metrics.enabled` | Enable metrics collection | `true` |
| `monitoring.tracing.enabled` | Enable distributed tracing | `false` |

### Autoscaling

| Parameter | Description | Default |
|-----------|-------------|---------|
| `autoscaling.enabled` | Enable HPA | `false` |
| `autoscaling.minReplicas` | Minimum replicas | `2` |
| `autoscaling.maxReplicas` | Maximum replicas | `10` |
| `autoscaling.targetCPUUtilizationPercentage` | Target CPU utilization | `70` |

## Upgrading

```bash
# Upgrade the release
helm upgrade brahminykite ./helm/brahminykite

# Upgrade with new values
helm upgrade brahminykite ./helm/brahminykite -f my-values.yaml
```

## Uninstalling

```bash
# Uninstall the release
helm uninstall brahminykite

# Uninstall and purge persistent volumes
helm uninstall brahminykite
kubectl delete pvc -l app.kubernetes.io/instance=brahminykite
```

## Testing

The chart includes test pods that can be run with:

```bash
helm test brahminykite
```

## Troubleshooting

### Check pod status
```bash
kubectl get pods -l app.kubernetes.io/instance=brahminykite
```

### View logs
```bash
# View logs for a specific service
kubectl logs -l app.kubernetes.io/component=empirical-service

# View API logs
kubectl logs -l app.kubernetes.io/component=api
```

### Access services locally

```bash
# Port-forward API
kubectl port-forward svc/brahminykite-api 8000:8000

# Port-forward Grafana
kubectl port-forward svc/brahminykite-grafana 3000:80

# Port-forward Prometheus
kubectl port-forward svc/brahminykite-prometheus-server 9090:80
```

## Values Examples

### Minimal production setup
```yaml
ingress:
  hosts:
    - host: api.example.com
      paths:
        - path: /
          pathType: Prefix
          service: api
          port: 8000

postgresql:
  auth:
    password: "your-secure-password"

api:
  env:
    CORS_ORIGINS: "https://app.example.com"
```

### Using external databases
```yaml
postgresql:
  enabled: false

externalPostgresql:
  host: postgres.example.com
  port: 5432
  username: brahminykite
  password: "your-password"
  database: brahminykite

redis:
  enabled: false

externalRedis:
  host: redis.example.com
  port: 6379
  password: "your-password"
```

## Support

For support, please open an issue in the [GitHub repository](https://github.com/brahminykite/brahminykite).