from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np
from PIL import Image

from app.masks import MotionComponent, build_motion_parse
from app.particles import ParticleField
from app.resize import resize_to_exact, resize_to_square_cover


@dataclass(frozen=True)
class _MotionLayer:
    name: str
    premult_bgr: np.ndarray
    alpha: np.ndarray
    center: Tuple[float, float]
    order: int
    angle_scale: float
    tx_scale: float
    ty_scale: float
    phase_offset: float


def _component_anchor(mask: np.ndarray, *, mode: str) -> Tuple[float, float] | None:
    pts = np.column_stack(np.nonzero(mask > 0.05))
    if pts.size == 0:
        return None
    ys = pts[:, 0].astype(np.float32)
    xs = pts[:, 1].astype(np.float32)
    if mode == "top":
        upper_cut = float(np.percentile(ys, 30))
        upper = ys <= upper_cut
        if upper.any():
            xs = xs[upper]
            ys = ys[upper]
        x_mid = float(np.median(xs))
        y_anchor = float(np.percentile(ys, 8))
    elif mode == "upper":
        x_mid = float(np.median(xs))
        y_anchor = float(np.percentile(ys, 18))
    else:
        x_mid = float(np.median(xs))
        y_anchor = float(np.percentile(ys, 32))
    return x_mid, y_anchor


def _extract_premultiplied_layer(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return bgr.astype(np.float32) * np.clip(alpha[..., None], 0.0, 1.0)


def _transform_layer(
    premult_bgr: np.ndarray,
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
    warped_layer = cv2.warpAffine(
        premult_bgr,
        matrix,
        (premult_bgr.shape[1], premult_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv2.warpAffine(
        alpha,
        matrix,
        (alpha.shape[1], alpha.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_layer, warped_alpha


def _composite_premult(base: np.ndarray, premult_layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha[..., None], 0.0, 1.0).astype(np.float32)
    out = base.astype(np.float32) * (1.0 - a) + premult_layer.astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _build_static_base(bgr: np.ndarray, hole_alpha: np.ndarray) -> np.ndarray:
    hole_mask = (hole_alpha > 0.16).astype(np.uint8)
    hole_ratio = float(hole_mask.mean())
    if hole_ratio <= 0.001 or hole_ratio >= 0.22:
        return bgr.copy()
    hole_mask = cv2.dilate(hole_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, hole_mask * 255, 3, cv2.INPAINT_TELEA)


def _prepare_layer_alpha(alpha: np.ndarray, *, gamma: float, blur_sigma: float) -> np.ndarray:
    out = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    if blur_sigma > 0.0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    out = np.power(np.clip(out, 0.0, 1.0), gamma).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _layer_from_component(
    bgr: np.ndarray,
    component: MotionComponent,
) -> list[_MotionLayer]:
    layers: list[_MotionLayer] = []

    hair_alpha = _prepare_layer_alpha(component.hair, gamma=0.95, blur_sigma=1.1)
    upper_alpha = _prepare_layer_alpha(component.garment_upper, gamma=1.00, blur_sigma=1.2)
    lower_alpha = _prepare_layer_alpha(component.garment_lower, gamma=1.05, blur_sigma=1.4)

    specs = [
        ("lower_cloth", lower_alpha, "middle", 10, 0.16, 1.16, 0.384, -0.25),
        ("upper_cloth", upper_alpha, "upper", 20, 0.272, 0.88, 0.144, 0.45),
        ("hair", hair_alpha, "top", 30, 0.576, 1.4, 0.192, 0.00),
    ]

    for name, alpha, anchor_mode, order, angle_scale, tx_scale, ty_scale, phase_offset in specs:
        if float(alpha.max()) < 0.06:
            continue
        anchor = _component_anchor(alpha, mode=anchor_mode)
        if anchor is None:
            continue
        layers.append(
            _MotionLayer(
                name=name,
                premult_bgr=_extract_premultiplied_layer(bgr, alpha),
                alpha=alpha,
                center=anchor,
                order=order,
                angle_scale=angle_scale,
                tx_scale=tx_scale,
                ty_scale=ty_scale,
                phase_offset=phase_offset,
            )
        )

    return layers


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
) -> Iterator[np.ndarray]:
    if fps <= 0 or fps > 60:
        raise ValueError("fps must be in 1..60")
    if motion_fps is not None and (motion_fps <= 0 or motion_fps > 60):
        raise ValueError("motion_fps must be in 1..60")
    if duration_sec <= 0.5 or duration_sec > 6.0:
        raise ValueError("duration_sec must be in 0.5..6.0")
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

    parsed = build_motion_parse(rgb)
    static_base = _build_static_base(bgr, parsed.base_hole)

    motion_layers: list[_MotionLayer] = []
    for component in parsed.components:
        motion_layers.extend(_layer_from_component(bgr, component))
    motion_layers.sort(key=lambda layer: layer.order)

    effective_motion_fps = fps if motion_fps is None else min(fps, motion_fps)
    motion_frames = max(4, int(round(duration_sec * effective_motion_fps)))

    particle_field = ParticleField.from_image(rgb, count=max(0, int(particles)), seed=2026)

    for t in range(motion_frames):
        phase = 2.0 * math.pi * (t / motion_frames)
        frame = static_base.copy()

        for layer in motion_layers:
            layer_phase = phase + layer.phase_offset
            sway = math.sin(layer_phase)
            lift = math.cos(layer_phase)
            flutter = math.sin((2.0 * phase) + layer.phase_offset) * 0.18

            warped_layer, warped_alpha = _transform_layer(
                layer.premult_bgr,
                layer.alpha,
                angle_deg=layer.angle_scale * strength * (sway + flutter),
                tx=layer.tx_scale * strength * (sway + 0.25 * flutter),
                ty=layer.ty_scale * strength * lift,
                center=layer.center,
            )
            frame = _composite_premult(frame, warped_layer, warped_alpha)

        if particle_field.count > 0:
            overlay = particle_field.render_frame_bgr(w, h, t=t, total_frames=motion_frames)
            frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0.0)

        yield frame
