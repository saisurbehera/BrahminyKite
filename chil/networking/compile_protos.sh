#!/bin/bash

# Compile protobuf files for networking layer

echo "Compiling consensus protocol buffers..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Compile the proto file
python -m grpc_tools.protoc \
    -I"$SCRIPT_DIR" \
    --python_out="$SCRIPT_DIR" \
    --grpc_python_out="$SCRIPT_DIR" \
    "$SCRIPT_DIR/consensus.proto"

# Fix imports in generated files
if [ -f "$SCRIPT_DIR/consensus_pb2_grpc.py" ]; then
    # Fix the import in the gRPC file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' 's/import consensus_pb2/from . import consensus_pb2/' "$SCRIPT_DIR/consensus_pb2_grpc.py"
    else
        # Linux
        sed -i 's/import consensus_pb2/from . import consensus_pb2/' "$SCRIPT_DIR/consensus_pb2_grpc.py"
    fi
fi

echo "Protocol buffers compiled successfully!"