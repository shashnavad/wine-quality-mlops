FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /tmp/data /tmp/models /tmp/metrics /tmp/processed /tmp/reports /tmp/deployment

# Copy source code and pipeline definitions
COPY src/ /app/src/
COPY pipelines/ /app/pipelines/

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "src/serving/app.py"] 