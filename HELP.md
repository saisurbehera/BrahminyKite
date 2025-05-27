# BrahminyKite Help & Scripts Guide

This guide provides helpful scripts and commands for working with the BrahminyKite project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Development Setup](#development-setup)
3. [Docker & Kubernetes](#docker--kubernetes)
4. [Service Management](#service-management)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/brahminykite/brahminykite.git
cd brahminykite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements-dev.txt
```

### Running Services Locally

```bash
# Start all services using the service manager
python -m chil.tools.service_manager

# Start individual services
python -m chil.tools.services.empirical_service
python -m chil.tools.services.contextual_service
# ... etc
```

## Development Setup

### Compile Protocol Buffers

```bash
# Compile all proto files
./scripts/compile_all_protos.sh

# Or compile specific modules
cd chil/tools && ./compile_protos.sh
cd chil/consensus && ./compile_protos.sh
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Service Configuration
GRPC_PORT_EMPIRICAL=50051
GRPC_PORT_CONTEXTUAL=50052
GRPC_PORT_CONSISTENCY=50053
GRPC_PORT_POWER_DYNAMICS=50054
GRPC_PORT_UTILITY=50055
GRPC_PORT_EVOLUTION=50056

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=brahminykite
POSTGRES_USER=brahminykite
POSTGRES_PASSWORD=your_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Consensus Network
CONSENSUS_NODE_ID=node-1
CONSENSUS_BIND_ADDRESS=0.0.0.0:7000
CONSENSUS_PEERS=node-2:7000,node-3:7000
```

## Docker & Kubernetes

### Docker Compose

```bash
# Start all services with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild images
docker-compose build

# Run with development overrides
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Kubernetes Deployment

```bash
# Deploy with raw manifests
kubectl apply -f k8s/base/

# Deploy with Kustomize
kubectl apply -k k8s/overlays/dev/    # Development
kubectl apply -k k8s/overlays/prod/   # Production

# Check deployment status
kubectl get pods -n brahminykite
kubectl get services -n brahminykite

# View logs
kubectl logs -n brahminykite -l app.kubernetes.io/name=empirical-service

# Port forward for local access
kubectl port-forward -n brahminykite svc/brahminykite-api 8000:8000
```

### Helm Deployment

```bash
# Install Helm chart
cd helm/brahminykite
helm dependency update
helm install brahminykite . -f values-dev.yaml --create-namespace -n brahminykite

# Upgrade deployment
helm upgrade brahminykite . -f values-prod.yaml -n brahminykite

# Check status
helm status brahminykite -n brahminykite

# Uninstall
helm uninstall brahminykite -n brahminykite
```

## Service Management

### Health Checks

```bash
# Check service health
curl -X POST localhost:50051/grpc.health.v1.Health/Check \
  -H 'content-type: application/grpc' \
  -H 'te: trailers'

# Check API health
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

### Monitoring

```bash
# Access Prometheus metrics
curl http://localhost:9090/metrics

# Port forward Grafana
kubectl port-forward -n brahminykite svc/grafana-service 3000:3000
# Default login: admin / grafana-admin-2024
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/unit/test_consensus.py
pytest tests/unit/test_framework.py

# Run with coverage
pytest --cov=chil --cov-report=html
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/ --integration

# Run end-to-end tests
python test_e2e_training.py
```

### Service Tests

```bash
# Test individual services
python test_services.py

# Test direct service connections
python test_services_direct.py

# Simple service test
python test_services_simple.py
```

## Troubleshooting

### Common Issues

#### 1. Proto Compilation Errors

```bash
# Install required tools
pip install grpcio-tools

# Ensure proto files are in correct location
ls chil/tools/protos/
ls chil/consensus/protos/
```

#### 2. Service Connection Issues

```bash
# Check if services are running
ps aux | grep "service.py"

# Check port availability
lsof -i :50051  # Check if port is in use

# Test gRPC connectivity
grpcurl -plaintext localhost:50051 list
```

#### 3. Docker Issues

```bash
# Clean up Docker resources
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache

# View container logs
docker logs <container_name>
```

#### 4. Kubernetes Issues

```bash
# Describe pod for errors
kubectl describe pod <pod_name> -n brahminykite

# Check events
kubectl get events -n brahminykite --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n brahminykite
```

### Debug Mode

```bash
# Run services in debug mode
LOG_LEVEL=DEBUG python -m chil.tools.service_manager

# Enable verbose gRPC logging
export GRPC_VERBOSITY=DEBUG
export GRPC_TRACE=all
```

### Performance Profiling

```bash
# Profile service performance
python -m cProfile -o profile.out python -m chil.tools.services.empirical_service

# Analyze profile
python -m pstats profile.out
```

## Useful Scripts

### Create Script Aliases

Add to your `.bashrc` or `.zshrc`:

```bash
# BrahminyKite aliases
alias bk-start='docker-compose up -d'
alias bk-stop='docker-compose down'
alias bk-logs='docker-compose logs -f'
alias bk-test='pytest'
alias bk-proto='./scripts/compile_all_protos.sh'
alias bk-clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null'
```

### Development Helper Script

Create `scripts/dev-setup.sh`:

```bash
#!/bin/bash
# Development setup helper

echo "Setting up BrahminyKite development environment..."

# Activate virtual environment
source venv/bin/activate

# Compile protos
./scripts/compile_all_protos.sh

# Start required services
docker-compose up -d postgres redis

# Wait for services
sleep 5

# Run database migrations
python scripts/init_db.py

echo "Development environment ready!"
echo "Start services with: python -m chil.tools.service_manager"
```

### Service Status Script

Create `scripts/check-status.sh`:

```bash
#!/bin/bash
# Check status of all BrahminyKite services

echo "Checking BrahminyKite services status..."
echo "======================================="

# Check Docker services
echo -e "\nDocker Services:"
docker-compose ps

# Check Kubernetes pods (if deployed)
if kubectl get ns brahminykite &>/dev/null; then
    echo -e "\nKubernetes Pods:"
    kubectl get pods -n brahminykite
fi

# Check local services
echo -e "\nLocal Service Ports:"
for port in 50051 50052 50053 50054 50055 50056 8000; do
    if lsof -i :$port &>/dev/null; then
        echo "Port $port: ✓ Active"
    else
        echo "Port $port: ✗ Inactive"
    fi
done
```

Make scripts executable:

```bash
chmod +x scripts/*.sh
```

## Additional Resources

- [Project Documentation](docs/)
- [API Documentation](http://localhost:8000/docs) (when API is running)
- [Consensus Architecture](chil/consensus/README.md)
- [Framework Details](docs/framework_organization.md)
- [Training Guide](docs/training_architecture.md)

For more help, please open an issue on GitHub or check the documentation.