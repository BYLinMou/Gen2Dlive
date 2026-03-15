from __future__ import annotations

import math
from typing import Iterator, Tuple

import numpy as np
import cv2
from PIL import Image

from app.masks import build_motion_masks
from app.particles import ParticleField
from app.resize import resize_to_exact, resize_to_square_cover


def _make_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return xs, ys


def _lowfreq_noise(h: int, w: int, rng: np.random.Generator, scale: int) -> np.ndarray:
    sh = max(2, h // scale)
    sw = max(2, w // scale)
    small = rng.random((sh, sw), dtype=np.float32)
    noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=6.0, sigmaY=6.0)
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    return noise


def _periodic_field(h: int, w: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1x = _lowfreq_noise(h, w, rng, scale=28)
    n2x = _lowfreq_noise(h, w, rng, scale=34)
    n1y = _lowfreq_noise(h, w, rng, scale=30)
    n2y = _lowfreq_noise(h, w, rng, scale=40)
    return (n1x + 0.7 * n2x).astype(np.float32), (n1y + 0.7 * n2y).astype(np.float32)


def generate_loop_frames_iter(
    pil_image: Image.Image,
    *,
    duration_sec: float,
    fps: int,
    motion_fps: int | None,
    width: int | None,
    height: int | None,
    size: int | None,
    strength: float,
    particles: int,
) -> "Iterator[np.ndarray]":
    if fps <= 0 or fps > 60:
        raise ValueError("fps must be in 1..60")
    if motion_fps is not None and (motion_fps <= 0 or motion_fps > 60):
        raise ValueError("motion_fps must be in 1..60")
    if duration_sec <= 0.5 or duration_sec > 30.0:
        raise ValueError("duration_sec must be in 0.5..30.0")
    if width is not None and (width < 256 or width > 2048):
        raise ValueError("width must be in 256..2048")
    if height is not None and (height < 256 or height > 2048):
        raise ValueError("height must be in 256..2048")
    if size is not None and (size < 256 or size > 2048):
        raise ValueError("size must be in 256..2048")
    if (width is None) != (height is None):
        raise ValueError("width and height must be provided together")
    if strength < 0.0 or strength > 3.0:
        raise ValueError("strength must be in 0..3")

    target_img = pil_image
    if width is not None and height is not None:
        target_img = resize_to_exact(pil_image, width=width, height=height)
    elif size is not None:
        target_img = resize_to_square_cover(pil_image, size=size)

    rgb = np.asarray(target_img).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    masks = build_motion_masks(rgb)
    hair = masks["hair"]
    sleeves = masks["sleeves"]
    cloth = masks["cloth_edge"]
    motion_mask = np.clip(0.9 * hair + 0.8 * sleeves + 0.6 * cloth, 0.0, 1.0).astype(np.float32)
    motion_mask = cv2.GaussianBlur(motion_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
    motion_mask = np.power(motion_mask, 1.08).astype(np.float32)

    motion_bin = (motion_mask > 0.04).astype(np.uint8)
    if motion_bin.any():
        x, y, bw, bh = cv2.boundingRect(motion_bin)
        pad = int(max(8, 0.02 * min(h, w)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        roi = (x0, y0, x1, y1)
    else:
        roi = None

    base_dx, base_dy = _periodic_field(h, w, seed=12345)
    grid_x, grid_y = _make_grid(h, w)

    effective_motion_fps = fps if motion_fps is None else min(fps, motion_fps)
    total_frames = int(round(duration_sec * fps))
    motion_frames = int(round(duration_sec * effective_motion_fps))
    if total_frames < 12:
        total_frames = 12
    if motion_frames < 12:
        motion_frames = 12

    particle_field = ParticleField.from_image(rgb, count=max(0, int(particles)), seed=2026)

    repeats = max(1, int(round(fps / effective_motion_fps)))
    produced = 0
    for t in range(motion_frames):
        phase = 2.0 * math.pi * (t / motion_frames)
        s1 = math.sin(phase)
        c1 = math.cos(phase)

        amp_px = 2.2 * strength
        dx = (0.70 * s1 * base_dx + 0.35 * c1 * base_dy) * amp_px
        dy = (0.50 * c1 * base_dy + 0.25 * s1 * base_dx) * (amp_px * 0.85)

        dx *= motion_mask
        dy *= motion_mask

        if roi is None:
            map_x = (grid_x + dx).astype(np.float32)
            map_y = (grid_y + dy).astype(np.float32)
            warped = cv2.remap(
                bgr,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        else:
            x0, y0, x1, y1 = roi
            map_x = (grid_x[y0:y1, x0:x1] + dx[y0:y1, x0:x1]).astype(np.float32)
            map_y = (grid_y[y0:y1, x0:x1] + dy[y0:y1, x0:x1]).astype(np.float32)
            warped = bgr.copy()
            warped_roi = cv2.remap(
                bgr,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            warped[y0:y1, x0:x1] = warped_roi

        if particle_field.count > 0:
            overlay = particle_field.render_frame_bgr(w, h, t=t, total_frames=total_frames)
            warped = cv2.addWeighted(warped, 1.0, overlay, 1.0, 0.0)

        for _ in range(repeats):
            if produced >= total_frames:
                break
            produced += 1
            yield warped

    while produced < total_frames:
        produced += 1
        yield warped
