"""Runtime configuration for the Q-Cash forecast service.

Everything is driven by environment variables so the same image can run in
development, staging and production without rebuilding.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SERVICE_DIR = APP_DIR.parent
DEFAULT_MODEL_PATH = SERVICE_DIR / "artifacts" / "service_model_bundle.joblib"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()

API_KEY = os.getenv("QCASH_API_KEY", "").strip()

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("QCASH_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
