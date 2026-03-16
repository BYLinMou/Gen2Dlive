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


def _soft_ramp(x: np.ndarray, start: float, end: float) -> np.ndarray:
    denom = max(1e-6, end - start)
    return np.clip((x - start) / denom, 0.0, 1.0).astype(np.float32)


def _component_anchor(mask: np.ndarray, *, mode: str) -> Tuple[float, float] | None:
    pts = np.column_stack(np.nonzero(mask > 0.05))
    if pts.size == 0:
        return None
    ys = pts[:, 0].astype(np.float32)
    xs = pts[:, 1].astype(np.float32)
    x_mid = float(xs.mean())
    if mode == "top":
        y_anchor = float(np.percentile(ys, 12))
    elif mode == "upper":
        y_anchor = float(np.percentile(ys, 22))
    else:
        y_anchor = float(np.percentile(ys, 35))
    return x_mid, y_anchor


def _transform_bgra(
    bgr: np.ndarray,
    alpha: np.ndarray,
    *,
    angle_deg: float,
    tx: float,
    ty: float,
    center: Tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    warped_bgr = cv2.warpAffine(
        bgr,
        matrix,
        (bgr.shape[1], bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_alpha = cv2.warpAffine(
        alpha,
        matrix,
        (alpha.shape[1], alpha.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_bgr, warped_alpha


def _composite(base: np.ndarray, layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha[..., None], 0.0, 1.0).astype(np.float32)
    return (base.astype(np.float32) * (1.0 - a) + layer.astype(np.float32) * a).astype(np.uint8)


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
    sleeve_left = masks["sleeve_left"]
    sleeve_right = masks["sleeve_right"]
    cloth = masks["cloth_edge"]
    motion_mask = np.clip(0.95 * hair + 1.05 * sleeve_left + 1.05 * sleeve_right + 1.20 * cloth, 0.0, 1.0).astype(np.float32)
    motion_mask = cv2.GaussianBlur(motion_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
    motion_mask = np.power(motion_mask, 0.90).astype(np.float32)

    grid_x, grid_y = _make_grid(h, w)
    xn = (grid_x / max(1.0, float(w - 1))) * 2.0 - 1.0
    yn = grid_y / max(1.0, float(h - 1))
    cx = np.abs(xn)

    hair_alpha = cv2.GaussianBlur((hair * _soft_ramp(yn, 0.18, 0.88) * (0.25 + 0.75 * _soft_ramp(cx, 0.10, 0.48))).astype(np.float32), (0, 0), sigmaX=4.0, sigmaY=4.0)
    sleeve_left_alpha = cv2.GaussianBlur((sleeve_left * _soft_ramp(yn, 0.24, 0.96)).astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0)
    sleeve_right_alpha = cv2.GaussianBlur((sleeve_right * _soft_ramp(yn, 0.24, 0.96)).astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0)
    cloth_alpha = cv2.GaussianBlur((cloth * _soft_ramp(yn, 0.40, 0.98)).astype(np.float32), (0, 0), sigmaX=6.0, sigmaY=6.0)

    hair_anchor = _component_anchor(hair_alpha, mode="top")
    sleeve_left_anchor = _component_anchor(sleeve_left_alpha, mode="upper")
    sleeve_right_anchor = _component_anchor(sleeve_right_alpha, mode="upper")
    cloth_anchor = _component_anchor(cloth_alpha, mode="middle")

    effective_motion_fps = fps if motion_fps is None else min(fps, motion_fps)
    total_frames = int(round(duration_sec * fps))
    motion_frames = int(round(duration_sec * effective_motion_fps))
    if total_frames < 12:
        total_frames = 12
    if motion_frames < 12:
        motion_frames = 12

    particle_field = ParticleField.from_image(rgb, count=max(0, int(particles)), seed=2026)

    for t in range(motion_frames):
        phase = 2.0 * math.pi * (t / motion_frames)
        s1 = math.sin(phase)
        c1 = math.cos(phase)
        warped = bgr.copy()

        if hair_anchor is not None:
            hair_layer, hair_layer_alpha = _transform_bgra(
                bgr,
                hair_alpha,
                angle_deg=3.6 * strength * s1,
                tx=10.0 * strength * s1,
                ty=2.0 * strength * c1,
                center=hair_anchor,
            )
            warped = _composite(warped, hair_layer, hair_layer_alpha)

        if sleeve_left_anchor is not None:
            left_layer, left_alpha = _transform_bgra(
                bgr,
                sleeve_left_alpha,
                angle_deg=-5.0 * strength * s1,
                tx=-8.0 * strength * s1,
                ty=6.0 * strength * c1,
                center=sleeve_left_anchor,
            )
            warped = _composite(warped, left_layer, left_alpha)

        if sleeve_right_anchor is not None:
            right_layer, right_alpha = _transform_bgra(
                bgr,
                sleeve_right_alpha,
                angle_deg=5.0 * strength * s1,
                tx=8.0 * strength * s1,
                ty=6.0 * strength * c1,
                center=sleeve_right_anchor,
            )
            warped = _composite(warped, right_layer, right_alpha)

        if cloth_anchor is not None:
            cloth_layer, cloth_layer_alpha = _transform_bgra(
                bgr,
                cloth_alpha,
                angle_deg=1.5 * strength * c1,
                tx=4.0 * strength * s1,
                ty=10.0 * strength * s1,
                center=cloth_anchor,
            )
            warped = _composite(warped, cloth_layer, cloth_layer_alpha)

        if particle_field.count > 0:
            overlay = particle_field.render_frame_bgr(w, h, t=t, total_frames=total_frames)
            warped = cv2.addWeighted(warped, 1.0, overlay, 1.0, 0.0)

        yield warped
