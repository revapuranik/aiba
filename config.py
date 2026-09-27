"""Environment-based settings for the application and vLLM connection."""

import os
from pathlib import Path


def _load_local_env_file():
    """Read simple KEY=value settings without executing the file as code."""
    env_file = Path(__file__).with_name(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        if name:
            os.environ.setdefault(name, value)


_load_local_env_file()


def _read_timeout():
    try:
        return float(os.getenv("VLLM_TIMEOUT_SECONDS", "45"))
    except ValueError:
        return 45.0


PLANNER_MODE = os.getenv("PLANNER_MODE", "llm").strip().lower()
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
VLLM_MODEL = os.getenv("VLLM_MODEL", "").strip()
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "").strip()
VLLM_TIMEOUT_SECONDS = _read_timeout()
