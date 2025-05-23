# Dockerfile for Chil Framework
FROM python:3.10-slim

LABEL maintainer="Chil Framework Team"
LABEL description="Unified philosophical verification and distributed consensus framework"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash chil
USER chil
WORKDIR /home/chil

# Copy and install Python dependencies
COPY --chown=chil:chil pyproject.toml .
COPY --chown=chil:chil README.md .

# Install the package
RUN pip install --user --upgrade pip setuptools wheel
RUN pip install --user -e .

# Copy application code
COPY --chown=chil:chil . .

# Install in development mode with all dependencies
RUN pip install --user -e .[dev]

# Expose port for API (if needed)
EXPOSE 8000

# Default command
CMD ["python", "-c", "import chil; print('Chil framework ready!')"]