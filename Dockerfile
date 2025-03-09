FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /tmp/data /tmp/models /tmp/metrics /tmp/processed /tmp/reports /tmp/deployment

# Copy source code and pipeline definitions
COPY src/ /app/src/
COPY pipelines/ /app/pipelines/

# Make component scripts executable
RUN chmod +x /app/src/components/*.py

# Set Python path
ENV PYTHONPATH=/app

# Set default entrypoint to python
ENTRYPOINT ["python"]

# Default to serving app (can be overridden)
CMD ["src/serving/app.py"] 