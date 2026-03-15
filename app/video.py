from __future__ import annotations

import os
import shutil
import subprocess
from typing import List

import cv2
import numpy as np


def _encode_with_ffmpeg(frames_bgr: List[np.ndarray], fps: int, out_path: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    h, w = frames_bgr[0].shape[:2]
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        out_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    try:
        for f in frames_bgr:
            proc.stdin.write(f.tobytes())
    finally:
        proc.stdin.close()
    return proc.wait() == 0 and os.path.exists(out_path)


def _encode_with_opencv(frames_bgr: List[np.ndarray], fps: int, out_path: str) -> None:
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter (mp4v). Install ffmpeg for better results.")
    try:
        for f in frames_bgr:
            writer.write(f)
    finally:
        writer.release()


def encode_mp4(*, frames_bgr: List[np.ndarray], fps: int, out_path: str) -> None:
    if not frames_bgr:
        raise ValueError("No frames to encode")
    ok = _encode_with_ffmpeg(frames_bgr, fps, out_path)
    if ok:
        return
    _encode_with_opencv(frames_bgr, fps, out_path)

