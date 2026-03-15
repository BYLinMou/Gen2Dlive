from __future__ import annotations

import shutil

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.service import generate_loop_mp4_to_tempfile

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@router.post("/animate")
def animate(
    image: UploadFile = File(...),
    duration_sec: float = Form(4.0),
    fps: int = Form(24),
    size: int = Form(768),
    strength: float = Form(1.0),
    particles: int = Form(18),
) -> FileResponse:
    out_path, tmp_dir = generate_loop_mp4_to_tempfile(
        image_file=image.file,
        duration_sec=duration_sec,
        fps=fps,
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
