# VisionRAG API Server
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (for Pillow, unstructured, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first for layer caching
COPY pyproject.toml ./

# Install dependencies only (no project code needed yet)
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" \
    "pymilvus>=2.4" "sentence-transformers>=3.0" "pillow>=10.4" \
    "python-multipart>=0.0.9" "celery[redis]>=5.4" "minio>=7.2" \
    "pydantic-settings>=2.4" "prometheus-client>=0.21" \
    "tritonclient[all]>=2.49"

# Copy application source
COPY src/ ./src/
COPY workers/ ./workers/
COPY scripts/ ./scripts/

# Install project in editable mode
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default: run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
