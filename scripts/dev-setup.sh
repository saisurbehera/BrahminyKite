#!/bin/bash
# Development environment setup script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}BrahminyKite Development Setup${NC}"
echo "=============================="

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Python 3 found${NC}"
fi

if ! command_exists docker; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
else
    echo -e "${GREEN}✓ Docker found${NC}"
fi

if ! command_exists docker-compose; then
    echo -e "${RED}✗ Docker Compose not found${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
else
    echo -e "${GREEN}✓ Docker Compose found${NC}"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "${PROJECT_ROOT}/venv" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "${PROJECT_ROOT}/venv"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source "${PROJECT_ROOT}/venv/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install package in development mode
echo -e "\n${YELLOW}Installing BrahminyKite in development mode...${NC}"
cd "${PROJECT_ROOT}"
pip install -e .

# Install additional dependencies
echo -e "\n${YELLOW}Installing additional dependencies...${NC}"
pip install grpcio-tools pytest pytest-asyncio pytest-cov

# Compile protocol buffers
echo -e "\n${YELLOW}Compiling protocol buffers...${NC}"
"${PROJECT_ROOT}/scripts/compile_all_protos.sh"

# Start infrastructure services
echo -e "\n${YELLOW}Starting infrastructure services...${NC}"
docker-compose up -d postgres redis

# Wait for services to be ready
echo -e "\n${YELLOW}Waiting for services to be ready...${NC}"
sleep 5

# Check service health
echo -e "\n${YELLOW}Checking service health...${NC}"
if docker-compose ps | grep -q "postgres.*Up"; then
    echo -e "${GREEN}✓ PostgreSQL is running${NC}"
else
    echo -e "${RED}✗ PostgreSQL failed to start${NC}"
fi

if docker-compose ps | grep -q "redis.*Up"; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${RED}✗ Redis failed to start${NC}"
fi

# Initialize database
echo -e "\n${YELLOW}Initializing database...${NC}"
if [ -f "${PROJECT_ROOT}/scripts/init_db.py" ]; then
    python "${PROJECT_ROOT}/scripts/init_db.py"
    echo -e "${GREEN}✓ Database initialized${NC}"
else
    echo -e "${YELLOW}⚠ No database initialization script found${NC}"
fi

# Create .env file if it doesn't exist
if [ ! -f "${PROJECT_ROOT}/.env" ]; then
    echo -e "\n${YELLOW}Creating .env file...${NC}"
    cat > "${PROJECT_ROOT}/.env" << EOF
# BrahminyKite Environment Configuration

# Service Ports
GRPC_PORT_EMPIRICAL=50051
GRPC_PORT_CONTEXTUAL=50052
GRPC_PORT_CONSISTENCY=50053
GRPC_PORT_POWER_DYNAMICS=50054
GRPC_PORT_UTILITY=50055
GRPC_PORT_EVOLUTION=50056

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=brahminykite
POSTGRES_USER=brahminykite
POSTGRES_PASSWORD=brahminykite-dev

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# Summary
echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}Development environment setup complete!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo -e "\nTo start the services:"
echo -e "  ${BLUE}python -m chil.tools.service_manager${NC}"
echo -e "\nTo run tests:"
echo -e "  ${BLUE}pytest${NC}"
echo -e "\nTo access the API documentation:"
echo -e "  ${BLUE}http://localhost:8000/docs${NC}"
echo -e "\nTo stop infrastructure services:"
echo -e "  ${BLUE}docker-compose down${NC}"