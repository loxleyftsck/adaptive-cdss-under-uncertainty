FROM python:3.9-slim

LABEL maintainer="heraldmsamueltheo@gmail.com"
LABEL description="RL-CDSS Training Container"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p logs/training models/checkpoints results/figures

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "scripts/experiments/run_baseline_comparison.py"]
