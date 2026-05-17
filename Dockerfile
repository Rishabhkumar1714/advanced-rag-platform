# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building native extensions (FAISS, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — minimal production image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Rishabh Kumar"
LABEL org.opencontainers.image.title="Advanced RAG Platform"
LABEL org.opencontainers.image.version="1.0.0"

# Non-root user for security
RUN groupadd -r raguser && useradd -r -g raguser -d /app raguser

WORKDIR /app

# System runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=raguser:raguser app/ ./app/
COPY --chown=raguser:raguser .env.example ./.env.example

# Create required directories
RUN mkdir -p data/raw data/processed logs && \
    chown -R raguser:raguser /app

USER raguser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Default command
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
