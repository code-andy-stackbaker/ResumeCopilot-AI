FROM --platform=linux/amd64 python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set pip timeout and upgrade pip early
RUN pip install --upgrade pip && \
    pip config set global.timeout 600

# Install PyTorch and other core dependencies
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    torchaudio==2.4.0+cpu \
    torchvision==0.19.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install pydantic and pydantic_core
RUN pip install --no-cache-dir \
    pydantic==2.11.3 \
    pydantic_core==2.33.1 

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy alembic config and scripts
COPY alembic.ini .
COPY alembic ./alembic

# Copy application code
COPY ./app ./app

# Create a non-root user and switch to it
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV HF_HOME=/tmp/huggingface_home_cache
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface_hub_model_cache
ENV TRANSFORMERS_CACHE=/tmp/transformers_model_cache

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]