from __future__ import annotations

import base64
import binascii
import shutil

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from app.auth import require_auth_if_configured
from app.service import generate_loop_mp4_from_bytes_to_tempfile, generate_loop_mp4_to_tempfile

router = APIRouter()

DATA_URI_PREFIX = "base64,"


class AnimateJsonRequest(BaseModel):
    image_base64: str
    duration_sec: float = 4.0
    fps: int = 30
    width: int | None = None
    height: int | None = None
    size: int | None = None
    strength: float = 1.0
    particles: int = 18


def _decode_base64_image(text: str) -> bytes:
    src = text.strip()
    lower_src = src.lower()
    if DATA_URI_PREFIX in lower_src:
        idx = lower_src.find(DATA_URI_PREFIX)
        src = src[idx + len(DATA_URI_PREFIX) :]
    try:
        return base64.b64decode(src, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc


@router.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@router.post("/animate")
def animate(
    _: None = Depends(require_auth_if_configured),
    image: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
    duration_sec: float = Form(4.0),
    fps: int = Form(30),
    width: int | None = Form(default=None),
    height: int | None = Form(default=None),
    size: int | None = Form(default=None),
    strength: float = Form(1.0),
    particles: int = Form(18),
) -> FileResponse:
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Provide either image file or image_base64")
    if image is not None and image_base64:
        raise HTTPException(status_code=400, detail="Use only one input: image or image_base64")

    if image is not None:
        out_path, tmp_dir = generate_loop_mp4_to_tempfile(
            image_file=image.file,
            duration_sec=duration_sec,
            fps=fps,
            width=width,
            height=height,
            size=size,
            strength=strength,
            particles=particles,
        )
    else:
        image_bytes = _decode_base64_image(image_base64 or "")
        out_path, tmp_dir = generate_loop_mp4_from_bytes_to_tempfile(
            image_bytes=image_bytes,
            duration_sec=duration_sec,
            fps=fps,
            width=width,
            height=height,
            size=size,
            strength=strength,
            particles=particles,
        )

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="loop.mp4",
        background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
    )


@router.post("/animate-json")
def animate_json(
    payload: AnimateJsonRequest,
    _: None = Depends(require_auth_if_configured),
) -> FileResponse:
    image_bytes = _decode_base64_image(payload.image_base64)
    out_path, tmp_dir = generate_loop_mp4_from_bytes_to_tempfile(
        image_bytes=image_bytes,
        duration_sec=payload.duration_sec,
        fps=payload.fps,
        width=payload.width,
        height=payload.height,
        size=payload.size,
        strength=payload.strength,
        particles=payload.particles,
    )
    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="loop.mp4",
        background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
    )
