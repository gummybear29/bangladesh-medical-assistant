FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and sample data
COPY src/ ./src/
COPY data/sample/ ./data/sample/
COPY evals/ ./evals/

# Pre-build index from sample data if not present
RUN python -c "from pathlib import Path; from src.ingest.indexer import index_labels; sample_dir = Path('data/sample'); index_labels(sample_dir)" || true

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
