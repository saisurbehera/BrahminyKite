#!/bin/bash
# Import Grafana dashboards for BrahminyKite

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
DASHBOARDS_DIR="${PROJECT_ROOT}/chil/monitoring/dashboards"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"

echo -e "${BLUE}Importing BrahminyKite Grafana Dashboards${NC}"
echo "=========================================="

# Check if Grafana is accessible
echo -e "\n${YELLOW}Checking Grafana connectivity...${NC}"
if curl -s -f "${GRAFANA_URL}/api/health" >/dev/null; then
    echo -e "${GREEN}✓ Grafana is accessible at ${GRAFANA_URL}${NC}"
else
    echo -e "${RED}✗ Cannot connect to Grafana at ${GRAFANA_URL}${NC}"
    echo "Please ensure Grafana is running and accessible."
    exit 1
fi

# Function to import a dashboard
import_dashboard() {
    local file=$1
    local name=$(basename "$file" .json)
    
    echo -e "\n${YELLOW}Importing dashboard: ${name}${NC}"
    
    # Read the dashboard JSON and wrap it in the required format
    local dashboard_json=$(cat "$file")
    local import_payload=$(jq -n --argjson dashboard "$dashboard_json" '{
        dashboard: $dashboard.dashboard,
        overwrite: true,
        inputs: []
    }')
    
    # Import the dashboard
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        -d "$import_payload" \
        "${GRAFANA_URL}/api/dashboards/db")
    
    # Check the response
    local status=$(echo "$response" | jq -r '.status // "error"')
    
    if [ "$status" = "success" ]; then
        local dashboard_url=$(echo "$response" | jq -r '.url')
        echo -e "${GREEN}✓ Dashboard imported successfully${NC}"
        echo -e "  URL: ${GRAFANA_URL}${dashboard_url}"
    else
        echo -e "${RED}✗ Failed to import dashboard${NC}"
        echo -e "  Response: $response"
    fi
}

# Check if jq is available
if ! command -v jq >/dev/null 2>&1; then
    echo -e "${RED}Error: jq is required but not installed${NC}"
    echo "Please install jq: https://stedolan.github.io/jq/download/"
    exit 1
fi

# Import all dashboards
echo -e "\n${YELLOW}Importing dashboards from ${DASHBOARDS_DIR}${NC}"

for dashboard_file in "${DASHBOARDS_DIR}"/*.json; do
    if [ -f "$dashboard_file" ]; then
        import_dashboard "$dashboard_file"
    fi
done

# Create data source if it doesn't exist
echo -e "\n${YELLOW}Checking Prometheus data source...${NC}"

# Check if Prometheus data source exists
datasource_exists=$(curl -s -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
    "${GRAFANA_URL}/api/datasources/name/Prometheus" | jq -r '.name // "not_found"')

if [ "$datasource_exists" = "not_found" ]; then
    echo -e "${YELLOW}Creating Prometheus data source...${NC}"
    
    datasource_payload='{
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://prometheus:9090",
        "access": "proxy",
        "isDefault": true,
        "basicAuth": false
    }'
    
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        -d "$datasource_payload" \
        "${GRAFANA_URL}/api/datasources")
    
    if echo "$response" | jq -e '.id' >/dev/null; then
        echo -e "${GREEN}✓ Prometheus data source created${NC}"
    else
        echo -e "${RED}✗ Failed to create Prometheus data source${NC}"
        echo -e "  Response: $response"
    fi
else
    echo -e "${GREEN}✓ Prometheus data source already exists${NC}"
fi

# Summary
echo -e "\n${GREEN}Dashboard import complete!${NC}"
echo -e "\nAccess your dashboards at:"
echo -e "  ${BLUE}${GRAFANA_URL}/dashboards${NC}"
echo -e "\nDefault login:"
echo -e "  Username: ${GRAFANA_USER}"
echo -e "  Password: ${GRAFANA_PASSWORD}"

echo -e "\n${YELLOW}Available dashboards:${NC}"
for dashboard_file in "${DASHBOARDS_DIR}"/*.json; do
    if [ -f "$dashboard_file" ]; then
        dashboard_title=$(jq -r '.dashboard.title' "$dashboard_file")
        echo -e "  • ${dashboard_title}"
    fi
done