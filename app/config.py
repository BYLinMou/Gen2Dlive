from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_env_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() != key:
            continue
        value = v.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return None


@lru_cache(maxsize=1)
def get_api_key() -> str:
    key = os.getenv("GEN2DLIVE_API_KEY")
    if key is not None:
        return key.strip()

    local_val = _read_env_value(PROJECT_ROOT / ".env.local", "GEN2DLIVE_API_KEY")
    if local_val is not None:
        return local_val.strip()

    env_val = _read_env_value(PROJECT_ROOT / ".env", "GEN2DLIVE_API_KEY")
    if env_val is not None:
        return env_val.strip()
    return ""


@lru_cache(maxsize=1)
def get_default_particles() -> int:
    raw = os.getenv("GEN2DLIVE_DEFAULT_PARTICLES")
    if raw is None:
        local_val = _read_env_value(PROJECT_ROOT / ".env.local", "GEN2DLIVE_DEFAULT_PARTICLES")
        raw = local_val if local_val is not None else _read_env_value(PROJECT_ROOT / ".env", "GEN2DLIVE_DEFAULT_PARTICLES")
    if raw is None:
        return 0
    try:
        value = int(str(raw).strip())
    except ValueError:
        return 0
    return max(0, min(200, value))


@lru_cache(maxsize=1)
def get_default_duration_sec() -> float:
    raw = os.getenv("GEN2DLIVE_DEFAULT_DURATION_SEC")
    if raw is None:
        local_val = _read_env_value(PROJECT_ROOT / ".env.local", "GEN2DLIVE_DEFAULT_DURATION_SEC")
        raw = local_val if local_val is not None else _read_env_value(PROJECT_ROOT / ".env", "GEN2DLIVE_DEFAULT_DURATION_SEC")
    if raw is None:
        return 20.0
    try:
        value = float(str(raw).strip())
    except ValueError:
        return 20.0
    return float(max(0.5, min(30.0, value)))


@lru_cache(maxsize=1)
def get_default_fps() -> int:
    raw = os.getenv("GEN2DLIVE_DEFAULT_FPS")
    if raw is None:
        local_val = _read_env_value(PROJECT_ROOT / ".env.local", "GEN2DLIVE_DEFAULT_FPS")
        raw = local_val if local_val is not None else _read_env_value(PROJECT_ROOT / ".env", "GEN2DLIVE_DEFAULT_FPS")
    if raw is None:
        return 30
    try:
        value = int(str(raw).strip())
    except ValueError:
        return 30
    return max(1, min(60, value))


@lru_cache(maxsize=1)
def get_default_strength() -> float:
    raw = os.getenv("GEN2DLIVE_DEFAULT_STRENGTH")
    if raw is None:
        local_val = _read_env_value(PROJECT_ROOT / ".env.local", "GEN2DLIVE_DEFAULT_STRENGTH")
        raw = local_val if local_val is not None else _read_env_value(PROJECT_ROOT / ".env", "GEN2DLIVE_DEFAULT_STRENGTH")
    if raw is None:
        return 2.0
    try:
        value = float(str(raw).strip())
    except ValueError:
        return 2.0
    return float(max(0.0, min(3.0, value)))


@lru_cache(maxsize=1)
def get_motion_fps() -> int:
    raw = os.getenv("GEN2DLIVE_MOTION_FPS")
    if raw is None:
        local_val = _read_env_value(PROJECT_ROOT / ".env.local", "GEN2DLIVE_MOTION_FPS")
        raw = local_val if local_val is not None else _read_env_value(PROJECT_ROOT / ".env", "GEN2DLIVE_MOTION_FPS")
    if raw is None:
        return 10
    try:
        value = int(str(raw).strip())
    except ValueError:
        return 10
    return max(1, min(60, value))


@lru_cache(maxsize=1)
def get_segmentation_backend() -> str:
    raw = os.getenv("GEN2DLIVE_SEGMENTATION_BACKEND")
    if raw is None:
        local_val = _read_env_value(PROJECT_ROOT / ".env.local", "GEN2DLIVE_SEGMENTATION_BACKEND")
        raw = local_val if local_val is not None else _read_env_value(PROJECT_ROOT / ".env", "GEN2DLIVE_SEGMENTATION_BACKEND")
    if raw is None:
        return "rembg"
    value = str(raw).strip().lower()
    if value in {"rembg", "heuristic"}:
        return value
    return "rembg"
