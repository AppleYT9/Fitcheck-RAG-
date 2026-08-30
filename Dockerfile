# ============================================================
# Stage 1: Build Frontend (Vite + React)
# ============================================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2: Python Backend & Production Server
# ============================================================
FROM python:3.11-slim
WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements with PyTorch CPU optimization (150MB instead of 2.5GB CUDA)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code and datasets
COPY . /app/

# Copy compiled frontend dist from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Environment configurations
EXPOSE 8000
ENV PORT=8000
ENV HOST=0.0.0.0
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Start FastAPI backend
CMD ["python", "run_api.py"]
