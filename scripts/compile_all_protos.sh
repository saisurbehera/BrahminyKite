#!/bin/bash
# Compile all protocol buffer files in the project

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

echo "Compiling all BrahminyKite protocol buffer files..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if grpcio-tools is installed
if ! python -c "import grpc_tools" 2>/dev/null; then
    echo -e "${RED}Error: grpcio-tools not installed${NC}"
    echo "Please run: pip install grpcio-tools"
    exit 1
fi

# Function to compile protos in a directory
compile_protos() {
    local dir=$1
    local name=$2
    
    echo -e "\n${YELLOW}Compiling ${name} protos...${NC}"
    
    if [ -f "${dir}/compile_protos.sh" ]; then
        cd "${dir}"
        ./compile_protos.sh
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ ${name} protos compiled successfully${NC}"
        else
            echo -e "${RED}✗ Failed to compile ${name} protos${NC}"
            return 1
        fi
        cd "${PROJECT_ROOT}"
    else
        echo -e "${YELLOW}⚠ No compile script found for ${name}${NC}"
    fi
}

# Compile all proto modules
compile_protos "${PROJECT_ROOT}/chil/tools" "Tools"
compile_protos "${PROJECT_ROOT}/chil/consensus" "Consensus"

echo -e "\n${GREEN}All protocol buffer compilations complete!${NC}"

# Create Python __init__.py files if needed
echo -e "\n${YELLOW}Ensuring __init__.py files exist...${NC}"

find "${PROJECT_ROOT}" -name "*_pb2*.py" -exec dirname {} \; | sort -u | while read dir; do
    if [ ! -f "${dir}/__init__.py" ]; then
        echo "# Auto-generated protobuf package" > "${dir}/__init__.py"
        echo -e "${GREEN}✓ Created __init__.py in ${dir}${NC}"
    fi
done

echo -e "\n${GREEN}Proto compilation setup complete!${NC}"