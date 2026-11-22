# Q-Cash decision-engine API
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QCASH_MODEL_PATH=/app/outputs/model.joblib

WORKDIR /app

# Dependencies first (better layer caching).
COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

# Application code + the trained artifact (build after running train.py / run_all.py).
COPY api.py engine.py common.py model_features.py config.yaml ./
COPY outputs/model.joblib ./outputs/model.joblib

EXPOSE 8000

# 1 worker is fine for a light model; scale with --workers on the VPS if needed.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
