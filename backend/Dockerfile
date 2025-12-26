# Genesis Brain - Backend Docker Image
# v3.8: Environment reproducibility

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/info')" || exit 1

# Run the application
CMD ["uvicorn", "main_genesis:app", "--host", "0.0.0.0", "--port", "8000"]
