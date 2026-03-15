from __future__ import annotations

import io
import os
import tempfile
from typing import BinaryIO

from PIL import Image

from app.animate import generate_loop_frames_iter
from app.config import get_motion_fps
from app.video import encode_mp4


def _read_image(image_file: BinaryIO) -> Image.Image:
    data = image_file.read()
    return _read_image_bytes(data)


def _read_image_bytes(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("Empty upload")
    return Image.open(io.BytesIO(data)).convert("RGB")


def generate_loop_mp4_to_tempfile(
    *,
    image_file: BinaryIO,
    duration_sec: float,
    fps: int,
    width: int | None,
    height: int | None,
    size: int | None,
    strength: float,
    particles: int,
) -> tuple[str, str]:
    pil_img = _read_image(image_file)
    motion_fps = get_motion_fps()
    frames = generate_loop_frames_iter(
        pil_img,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        size=size,
        strength=strength,
        particles=particles,
        motion_fps=motion_fps,
    )
    tmp_dir = tempfile.mkdtemp(prefix="gen2dlive_")
    out_path = os.path.join(tmp_dir, "loop.mp4")
    encode_mp4(frames_bgr=frames, fps=fps, input_fps=motion_fps, out_path=out_path)
    return out_path, tmp_dir


def generate_loop_mp4_from_bytes_to_tempfile(
    *,
    image_bytes: bytes,
    duration_sec: float,
    fps: int,
    width: int | None,
    height: int | None,
    size: int | None,
    strength: float,
    particles: int,
) -> tuple[str, str]:
    pil_img = _read_image_bytes(image_bytes)
    motion_fps = get_motion_fps()
    frames = generate_loop_frames_iter(
        pil_img,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        size=size,
        strength=strength,
        particles=particles,
        motion_fps=motion_fps,
    )
    tmp_dir = tempfile.mkdtemp(prefix="gen2dlive_")
    out_path = os.path.join(tmp_dir, "loop.mp4")
    encode_mp4(frames_bgr=frames, fps=fps, input_fps=motion_fps, out_path=out_path)
    return out_path, tmp_dir
