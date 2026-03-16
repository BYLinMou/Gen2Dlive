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


def _read_setting(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            return value
    for env_name in (".env.local", ".env"):
        env_path = PROJECT_ROOT / env_name
        for key in keys:
            value = _read_env_value(env_path, key)
            if value is not None:
                return value
    return None


@lru_cache(maxsize=1)
def get_api_key() -> str:
    value = _read_setting("GEN2DLIVE_API_KEY")
    return "" if value is None else value.strip()


@lru_cache(maxsize=1)
def get_default_particles() -> int:
    raw = _read_setting("GEN2DLIVE_DEFAULT_PARTICLES")
    if raw is None:
        return 0
    try:
        value = int(str(raw).strip())
    except ValueError:
        return 0
    return max(0, min(200, value))


@lru_cache(maxsize=1)
def get_default_duration_sec() -> float:
    raw = _read_setting("GEN2DLIVE_DEFAULT_CYCLE_SEC", "GEN2DLIVE_DEFAULT_DURATION_SEC")
    if raw is None:
        return 3.0
    try:
        value = float(str(raw).strip())
    except ValueError:
        return 3.0
    return float(max(0.5, min(6.0, value)))


@lru_cache(maxsize=1)
def get_default_fps() -> int:
    raw = _read_setting("GEN2DLIVE_DEFAULT_FPS")
    if raw is None:
        return 30
    try:
        value = int(str(raw).strip())
    except ValueError:
        return 30
    return max(1, min(60, value))


@lru_cache(maxsize=1)
def get_default_strength() -> float:
    raw = _read_setting("GEN2DLIVE_DEFAULT_STRENGTH")
    if raw is None:
        return 2.0
    try:
        value = float(str(raw).strip())
    except ValueError:
        return 2.0
    return float(max(0.0, min(3.0, value)))


@lru_cache(maxsize=1)
def get_motion_fps() -> int:
    raw = _read_setting("GEN2DLIVE_MOTION_FPS")
    if raw is None:
        return 10
    try:
        value = int(str(raw).strip())
    except ValueError:
        return 10
    return max(1, min(60, value))


@lru_cache(maxsize=1)
def get_segmentation_backend() -> str:
    raw = _read_setting("GEN2DLIVE_SEGMENTATION_BACKEND")
    if raw is None:
        return "rembg"
    value = str(raw).strip().lower()
    if value in {"rembg", "heuristic"}:
        return value
    return "rembg"


@lru_cache(maxsize=1)
def get_segmentation_model() -> str:
    raw = _read_setting("GEN2DLIVE_SEGMENTATION_MODEL")
    if raw is None:
        return "isnet-anime"
    value = str(raw).strip().lower()
    if value in {"isnet-anime", "u2net_human_seg", "u2net"}:
        return value
    return "isnet-anime"


@lru_cache(maxsize=1)
def get_job_workers() -> int:
    raw = _read_setting("GEN2DLIVE_JOB_WORKERS")
    if raw is None:
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count))
    try:
        value = int(str(raw).strip())
    except ValueError:
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count))
    return max(1, min(16, value))
