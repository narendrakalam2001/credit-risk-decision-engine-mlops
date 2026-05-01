# ============================================================
# DOCKERFILE — Credit Risk ML System
# ============================================================
# Multi-stage approach:
#   Stage 1 (builder) — install all dependencies
#   Stage 2 (runtime) — lean final image
# ============================================================

# ── Base image ────────────────────────────────────────────────
FROM python:3.10.13-slim AS builder

# Prevent Python from writing .pyc files + unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed by LightGBM / XGBoost / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (API-only — leaner image)
COPY requirements_api.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt


# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.10.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libgomp1 required at runtime by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY src/          ./src/
COPY serving/      ./serving/
COPY services/     ./services/
COPY risk_models/  ./risk_models/
COPY logs/         ./logs/

# Expose API port
EXPOSE 8000

# Health check — hits /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Start FastAPI with uvicorn
CMD ["uvicorn", "serving.credit_risk_api:app", "--host", "0.0.0.0", "--port", "8000"]
