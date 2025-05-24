#!/bin/bash
# Compile gRPC protocol buffers

echo "Compiling protocol buffers..."

# Create output directory
mkdir -p protos

# Compile protos
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    protos/tools.proto

# Fix imports in generated files
sed -i '' 's/import tools_pb2/from . import tools_pb2/g' protos/tools_pb2_grpc.py

echo "✓ Protocol buffers compiled successfully"

# Make files importable
touch protos/__init__.py

echo "✓ Setup complete"