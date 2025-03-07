FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/serving/app.py .
COPY models/model.joblib /models/

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"] 