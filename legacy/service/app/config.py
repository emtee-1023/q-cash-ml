"""Runtime configuration for the Q-Cash assessment service.

Everything is driven by environment variables so the same image can run in
development, staging and production without rebuilding.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SERVICE_DIR = APP_DIR.parent
REPO_DIR = SERVICE_DIR.parent

DEFAULT_MODEL_PATH = SERVICE_DIR / "artifacts" / "service_model_bundle.joblib"
# Fallback so the service runs locally and in tests without copying the trained
# artifact into service/artifacts (which is gitignored). The Docker image still
# copies the artifact and points MODEL_PATH at /app/artifacts explicitly.
REPO_MODEL_PATH = REPO_DIR / "qcash_synthetic_v1" / "models" / "service_model_bundle.joblib"

_model_env = os.getenv("MODEL_PATH", "").strip()
if _model_env:
    MODEL_PATH = Path(_model_env).resolve()
elif DEFAULT_MODEL_PATH.exists():
    MODEL_PATH = DEFAULT_MODEL_PATH
else:
    MODEL_PATH = REPO_MODEL_PATH

DEFAULT_POLICY_PATH = SERVICE_DIR / "policies" / "era_policy_v1.json"
POLICY_PATH = Path(
    os.getenv("QCASH_POLICY_PATH", str(DEFAULT_POLICY_PATH))
).resolve()

API_KEY = os.getenv("QCASH_API_KEY", "").strip()

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("QCASH_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

LOG_LEVEL = os.getenv("QCASH_LOG_LEVEL", "INFO").strip().upper()
