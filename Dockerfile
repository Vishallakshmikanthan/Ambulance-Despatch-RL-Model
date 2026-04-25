FROM python:3.11-slim

WORKDIR /app

# Install system dependencies in a separate layer for Docker cache efficiency
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS (more stable than distro nodejs)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before application code for cache efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code last
COPY . .

# Build Next.js dashboard — next.config.js sets output:'export' + distDir:'dist'
# || true ensures build failures don't break the Docker image (backend works standalone)
RUN if [ -d "frontend" ]; then \
    cd frontend && \
    npm install --legacy-peer-deps --silent 2>/dev/null || true && \
    npm run build 2>/dev/null || true && \
    cd ..; \
    fi

ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
# HF_TOKEN has no default — must be provided at runtime
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE="true"
ENV PORT=7860
ENV WORKERS=1
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers ${WORKERS:-1}"]
