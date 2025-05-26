#!/bin/bash
# Script to compile consensus protocol buffer files

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROTO_DIR="${SCRIPT_DIR}/protos"
OUTPUT_DIR="${PROTO_DIR}"

echo "Compiling consensus protobuf files..."

# Check if grpcio-tools is installed
if ! python -c "import grpc_tools" 2>/dev/null; then
    echo "Error: grpcio-tools not installed. Run: pip install grpcio-tools"
    exit 1
fi

# Compile proto files
python -m grpc_tools.protoc \
    --proto_path="${PROTO_DIR}" \
    --python_out="${OUTPUT_DIR}" \
    --grpc_python_out="${OUTPUT_DIR}" \
    "${PROTO_DIR}/consensus.proto"

# Fix imports in generated files
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' 's/import consensus_pb2/from . import consensus_pb2/g' "${OUTPUT_DIR}/consensus_pb2_grpc.py"
else
    # Linux
    sed -i 's/import consensus_pb2/from . import consensus_pb2/g' "${OUTPUT_DIR}/consensus_pb2_grpc.py"
fi

echo "Consensus protobuf compilation complete!"

# Create __init__.py if it doesn't exist
if [ ! -f "${OUTPUT_DIR}/__init__.py" ]; then
    echo "# Auto-generated protobuf package" > "${OUTPUT_DIR}/__init__.py"
fi