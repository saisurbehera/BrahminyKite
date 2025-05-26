#!/bin/bash
# Check status of all BrahminyKite services

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}BrahminyKite Service Status${NC}"
echo "==========================="
echo -e "Time: $(date)"

# Function to check if port is in use
check_port() {
    local port=$1
    local service=$2
    if lsof -i :$port >/dev/null 2>&1; then
        echo -e "${GREEN}✓ ${service} (port ${port}): Active${NC}"
        return 0
    else
        echo -e "${RED}✗ ${service} (port ${port}): Inactive${NC}"
        return 1
    fi
}

# Check Docker services
echo -e "\n${YELLOW}Docker Services:${NC}"
if command -v docker-compose >/dev/null 2>&1; then
    cd "${PROJECT_ROOT}"
    docker-compose ps --services --filter "status=running" | while read service; do
        echo -e "${GREEN}✓ ${service}: Running${NC}"
    done
    docker-compose ps --services --filter "status=exited" | while read service; do
        if [ ! -z "$service" ]; then
            echo -e "${RED}✗ ${service}: Stopped${NC}"
        fi
    done
else
    echo -e "${YELLOW}Docker Compose not available${NC}"
fi

# Check Kubernetes services (if kubectl is available)
echo -e "\n${YELLOW}Kubernetes Services:${NC}"
if command -v kubectl >/dev/null 2>&1; then
    if kubectl get ns brahminykite >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Namespace 'brahminykite' exists${NC}"
        
        # Get pod status
        echo -e "\n  Pods:"
        kubectl get pods -n brahminykite --no-headers | while read line; do
            name=$(echo $line | awk '{print $1}')
            ready=$(echo $line | awk '{print $2}')
            status=$(echo $line | awk '{print $3}')
            
            if [ "$status" = "Running" ]; then
                echo -e "  ${GREEN}✓ ${name}: ${status} (${ready})${NC}"
            else
                echo -e "  ${RED}✗ ${name}: ${status}${NC}"
            fi
        done
    else
        echo -e "${YELLOW}Kubernetes namespace 'brahminykite' not found${NC}"
    fi
else
    echo -e "${YELLOW}kubectl not available${NC}"
fi

# Check local service ports
echo -e "\n${YELLOW}Local Service Ports:${NC}"
check_port 50051 "Empirical Service"
check_port 50052 "Contextual Service"
check_port 50053 "Consistency Service"
check_port 50054 "Power Dynamics Service"
check_port 50055 "Utility Service"
check_port 50056 "Evolution Service"
check_port 8000 "FastAPI"
check_port 5432 "PostgreSQL"
check_port 6379 "Redis"
check_port 9090 "Prometheus"
check_port 3000 "Grafana"

# Check gRPC health if grpcurl is available
if command -v grpcurl >/dev/null 2>&1; then
    echo -e "\n${YELLOW}gRPC Health Checks:${NC}"
    for port in 50051 50052 50053 50054 50055 50056; do
        if lsof -i :$port >/dev/null 2>&1; then
            if grpcurl -plaintext localhost:$port grpc.health.v1.Health/Check >/dev/null 2>&1; then
                echo -e "${GREEN}✓ Service on port ${port}: Healthy${NC}"
            else
                echo -e "${RED}✗ Service on port ${port}: Unhealthy${NC}"
            fi
        fi
    done
else
    echo -e "\n${YELLOW}Install grpcurl for gRPC health checks${NC}"
fi

# Check API health
echo -e "\n${YELLOW}API Health Check:${NC}"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
    echo -e "${GREEN}✓ API: Healthy${NC}"
    
    # Get API details
    api_info=$(curl -s http://localhost:8000/docs | grep -o "<title>.*</title>" | sed 's/<[^>]*>//g')
    if [ ! -z "$api_info" ]; then
        echo -e "  ${BLUE}${api_info}${NC}"
    fi
else
    echo -e "${RED}✗ API: Not responding${NC}"
fi

# System resource usage
echo -e "\n${YELLOW}System Resources:${NC}"
if command -v docker >/dev/null 2>&1; then
    echo -e "\nDocker Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep -E "(brahminykite|CONTAINER)"
fi

# Summary
echo -e "\n${YELLOW}Summary:${NC}"
active_services=0
total_services=11  # 6 gRPC + API + PostgreSQL + Redis + Prometheus + Grafana

for port in 50051 50052 50053 50054 50055 50056 8000 5432 6379 9090 3000; do
    if lsof -i :$port >/dev/null 2>&1; then
        ((active_services++))
    fi
done

echo -e "Active services: ${active_services}/${total_services}"

if [ $active_services -eq $total_services ]; then
    echo -e "${GREEN}All services are running!${NC}"
elif [ $active_services -eq 0 ]; then
    echo -e "${RED}No services are running!${NC}"
else
    echo -e "${YELLOW}Some services are not running${NC}"
fi

# Helpful commands
echo -e "\n${YELLOW}Helpful Commands:${NC}"
echo -e "Start all services:     ${BLUE}docker-compose up -d${NC}"
echo -e "Start local services:   ${BLUE}python -m chil.tools.service_manager${NC}"
echo -e "View logs:             ${BLUE}docker-compose logs -f${NC}"
echo -e "Stop all services:     ${BLUE}docker-compose down${NC}"