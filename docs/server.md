# BrahminyKite Server Architecture

Complete deployment and infrastructure guide for the BrahminyKite text environment and tools ecosystem.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
│                   (nginx/HAProxy)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                FastAPI Server                               │
│               (chil.env.api)                               │
│            Port 8000 (Multiple Workers)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 gRPC Tool Services                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Empirical   │ │ Contextual  │ │ Consistency │          │
│  │ Port 50051  │ │ Port 50052  │ │ Port 50053  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Power Dyn.  │ │ Utility     │ │ Evolution   │          │
│  │ Port 50054  │ │ Port 50055  │ │ Port 50056  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 Data Layer                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ DuckDB      │ │ SQLite      │ │ FAISS       │          │
│  │ (Facts)     │ │ (Patterns)  │ │ (Vectors)   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Oxigraph    │ │ LMDB        │ │ Models      │          │
│  │ (RDF)       │ │ (Cache)     │ │ (XGBoost)   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Options

### 1. Development Setup

Single machine deployment for development and testing.

```bash
# Start all services
python -m chil.tools.service_manager

# Start FastAPI server
python -m chil.env.run_server --reload
```

### 2. Production Deployment

#### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  # FastAPI Application
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - TOOLS_CONFIG=/app/config/production.yaml
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    depends_on:
      - empirical
      - contextual
      - consistency
      - power
      - utility
      - evolution
    restart: unless-stopped

  # Tool Services
  empirical:
    build: 
      context: .
      dockerfile: Dockerfile.tools
    command: python -m chil.tools.services.empirical_service
    ports:
      - "50051:50051"
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  contextual:
    build:
      context: .
      dockerfile: Dockerfile.tools
    command: python -m chil.tools.services.contextual_service
    ports:
      - "50052:50052"
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  consistency:
    build:
      context: .
      dockerfile: Dockerfile.tools
    command: python -m chil.tools.services.consistency_service
    ports:
      - "50053:50053"
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  power:
    build:
      context: .
      dockerfile: Dockerfile.tools
    command: python -m chil.tools.services.power_dynamics_service
    ports:
      - "50054:50054"
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  utility:
    build:
      context: .
      dockerfile: Dockerfile.tools
    command: python -m chil.tools.services.utility_service
    ports:
      - "50055:50055"
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  evolution:
    build:
      context: .
      dockerfile: Dockerfile.tools
    command: python -m chil.tools.services.evolution_service
    ports:
      - "50056:50056"
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  # Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-data:
```

#### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: brahminy-kite-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: brahminy-kite-api
  template:
    metadata:
      labels:
        app: brahminy-kite-api
    spec:
      containers:
      - name: api
        image: brahminy-kite:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: brahminy-kite-api-service
spec:
  selector:
    app: brahminy-kite-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 3. Cloud Deployment

#### AWS ECS/Fargate

```json
{
  "family": "brahminy-kite",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-registry/brahminy-kite:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/brahminy-kite",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## Configuration Management

### Environment-Specific Configs

#### Development (`config/development.yaml`)
```yaml
services:
  empirical:
    port: 50051
    max_workers: 2
    duckdb_path: "./data/dev_facts.db"
    
  contextual:
    port: 50052
    max_workers: 2
    spacy_model: "en_core_web_sm"

api:
  host: "127.0.0.1"
  port: 8000
  workers: 1
  reload: true
  log_level: "DEBUG"

performance:
  cache:
    size: 1000
    ttl_seconds: 300
```

#### Production (`config/production.yaml`)
```yaml
services:
  empirical:
    port: 50051
    max_workers: 10
    duckdb_path: "/data/facts.db"
    cache_size: 50000
    
  contextual:
    port: 50052
    max_workers: 8
    spacy_model: "en_core_web_lg"
    faiss_index_path: "/data/embeddings.index"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  log_level: "INFO"

performance:
  cache:
    size: 100000
    ttl_seconds: 3600
    
  batching:
    enabled: true
    max_batch_size: 500
    batch_timeout_ms: 100
```

## Performance Optimization

### 1. Horizontal Scaling

```bash
# Scale API servers
docker-compose up --scale api=4

# Scale individual tool services
docker-compose up --scale empirical=2 --scale contextual=2
```

### 2. Load Balancing

#### Nginx Configuration
```nginx
upstream api_backend {
    least_conn;
    server api:8000 weight=1 max_fails=3 fail_timeout=30s;
    server api:8001 weight=1 max_fails=3 fail_timeout=30s;
    server api:8002 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://api_backend/health;
        access_log off;
    }
}
```

### 3. Caching Strategy

#### Redis Integration
```python
# config/cache.py
import redis
from typing import Any, Optional

class RedisCache:
    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
    
    def get(self, key: str) -> Optional[Any]:
        return self.client.get(key)
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        self.client.setex(key, ttl, value)
    
    def delete(self, key: str):
        self.client.delete(key)
```

## Monitoring and Observability

### 1. Health Checks

```python
# health_check.py
async def health_check():
    services = {
        'empirical': check_service_health('localhost:50051'),
        'contextual': check_service_health('localhost:50052'),
        'consistency': check_service_health('localhost:50053'),
        'power': check_service_health('localhost:50054'),
        'utility': check_service_health('localhost:50055'),
        'evolution': check_service_health('localhost:50056')
    }
    
    return {
        'status': 'healthy' if all(services.values()) else 'unhealthy',
        'services': services,
        'timestamp': datetime.now().isoformat()
    }
```

### 2. Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_SESSIONS = Gauge('active_sessions_total', 'Number of active sessions')

# Framework metrics
FRAMEWORK_CALLS = Counter('framework_calls_total', 'Framework verification calls', ['framework'])
FRAMEWORK_DURATION = Histogram('framework_duration_seconds', 'Framework call duration', ['framework'])
FRAMEWORK_ERRORS = Counter('framework_errors_total', 'Framework errors', ['framework', 'error_type'])
```

### 3. Logging

```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'brahminy_kite.log',
            'level': 'DEBUG',
            'formatter': 'detailed',
        },
    },
    'loggers': {
        'chil': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
}
```

## Security

### 1. API Authentication

```python
# auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Implement your token verification logic
    if not verify_jwt_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token
```

### 2. Rate Limiting

```python
# rate_limit.py
from fastapi import Request, HTTPException
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
    
    async def check_rate_limit(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.window
        ]
        
        if len(self.requests[client_ip]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        self.requests[client_ip].append(now)
```

## Backup and Recovery

### 1. Data Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$DATE"

mkdir -p $BACKUP_DIR

# Backup databases
cp data/facts.db $BACKUP_DIR/
cp data/patterns.db $BACKUP_DIR/
cp -r data/knowledge.rdf $BACKUP_DIR/
cp -r data/embeddings.index $BACKUP_DIR/

# Backup models
tar -czf $BACKUP_DIR/models.tar.gz models/

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/$DATE
```

### 2. Disaster Recovery

```bash
#!/bin/bash
# restore.sh

BACKUP_DATE=$1
BACKUP_DIR="/backups/$BACKUP_DATE"

# Stop services
docker-compose down

# Restore data
cp $BACKUP_DIR/facts.db data/
cp $BACKUP_DIR/patterns.db data/
cp -r $BACKUP_DIR/knowledge.rdf data/
cp -r $BACKUP_DIR/embeddings.index data/

# Restore models
tar -xzf $BACKUP_DIR/models.tar.gz

# Restart services
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. **Service Discovery Failures**
   - Check gRPC service health
   - Verify network connectivity
   - Check service configuration

2. **High Memory Usage**
   - Monitor FAISS index size
   - Check for memory leaks in caching
   - Optimize model loading

3. **Slow Response Times**
   - Enable request batching
   - Increase cache sizes
   - Scale horizontally

### Debug Commands

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f empirical

# Monitor resource usage
docker stats

# Test connectivity
curl http://localhost:8000/health
grpcurl -plaintext localhost:50051 list
```