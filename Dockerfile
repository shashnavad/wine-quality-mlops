# Use Python 3.9 as base
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and install Argo executor
RUN mkdir -p /var/run/argo \
    && curl -L -o /var/run/argo/argoexec https://github.com/argoproj/argo-workflows/releases/download/v3.4.11/argoexec-linux-amd64 \
    && chmod +x /var/run/argo/argoexec

# Create directories for data, models, and metrics
RUN mkdir -p /tmp/processed /tmp/models /tmp/metrics

# Set working directory
WORKDIR /app

# Copy pipeline definitions
COPY pipelines/ /app/pipelines/

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python"] 