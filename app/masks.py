from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

import cv2
import numpy as np

from app.config import get_segmentation_backend, get_segmentation_model


@dataclass(frozen=True)
class MotionComponent:
    foreground: np.ndarray
    face_block: np.ndarray
    hair: np.ndarray
    garment_upper: np.ndarray
    garment_lower: np.ndarray


@dataclass(frozen=True)
class MotionParse:
    foreground: np.ndarray
    face_block: np.ndarray
    hair: np.ndarray
    garment_upper: np.ndarray
    garment_lower: np.ndarray
    base_hole: np.ndarray
    components: tuple[MotionComponent, ...]


def _soft_clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _soft_ramp(x: np.ndarray, start: float, end: float) -> np.ndarray:
    denom = max(1e-6, end - start)
    return np.clip((x - start) / denom, 0.0, 1.0).astype(np.float32)


def _edge_weight(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=1.8, sigmaY=1.8)
    return _soft_clip01(mag)


def _ellipse_kernel(radius: int) -> np.ndarray:
    size = max(3, radius * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _background_mask(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    ).astype(np.float32)
    if border.size == 0:
        return np.zeros((h, w), dtype=bool)
    med = np.median(border, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - med[None, None, :], axis=-1)
    border_dist = np.linalg.norm(border - med[None, :], axis=-1)
    m = np.median(border_dist)
    mad = np.median(np.abs(border_dist - m)) + 1e-6
    thresh = m + 3.0 * mad + 8.0
    bg = dist <= thresh
    bg = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    bg = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    return bg


def _grabcut_foreground(rgb: np.ndarray) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    if h < 64 or w < 64:
        return None
    rect = (
        int(w * 0.05),
        int(h * 0.05),
        int(w * 0.90),
        int(h * 0.90),
    )
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    ratio = float(fg.mean())
    if ratio < 0.03 or ratio > 0.97:
        return None
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    return fg


@lru_cache(maxsize=3)
def _get_rembg_session(model_name: str):
    from rembg import new_session

    return new_session(model_name)


def _rembg_foreground(rgb: np.ndarray) -> np.ndarray | None:
    if get_segmentation_backend() != "rembg":
        return None
    try:
        from rembg import remove
    except Exception:
        return None

    model_candidates = [get_segmentation_model(), "u2net_human_seg", "u2net"]
    for model_name in model_candidates:
        try:
            session = _get_rembg_session(model_name)
            mask = remove(rgb, session=session, only_mask=True, post_process_mask=True)
        except Exception:
            continue
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]
        fg = mask.astype(np.float32) / 255.0
        fg = cv2.GaussianBlur(fg, (0, 0), sigmaX=1.2, sigmaY=1.2)
        fg = fg > 0.42
        ratio = float(fg.mean())
        if 0.02 < ratio < 0.97:
            return fg
    return None


def _refine_foreground_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    area_threshold = max(96, int(0.0012 * h * w))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < area_threshold:
            continue
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        ww = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        hh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if hh < max(24, int(0.08 * h)):
            continue
        if ww < max(16, int(0.02 * w)):
            continue
        keep |= labels == label_idx
    if not keep.any():
        keep = mask > 0
    radius = max(1, int(round(0.004 * min(h, w))))
    kernel = _ellipse_kernel(radius)
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return keep


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    fg = _rembg_foreground(rgb)
    if fg is None:
        fg = _grabcut_foreground(rgb)
    if fg is None:
        fg = ~_background_mask(rgb)
    return _refine_foreground_mask(fg)


def _distance_weights(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    sharp_scale = max(2.0, 0.022 * min(h, w))
    soft_scale = max(7.0, 0.09 * min(h, w))
    sharp = np.exp(-((dist / sharp_scale) ** 2)).astype(np.float32)
    soft = np.exp(-((dist / soft_scale) ** 2)).astype(np.float32)
    return _soft_clip01(sharp), _soft_clip01(soft)


def _smooth_signal(values: np.ndarray, sigma: float) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    smoothed = cv2.GaussianBlur(values.astype(np.float32)[:, None], (0, 0), sigmaX=0.0, sigmaY=sigma)
    return smoothed[:, 0]


def _component_masks(fg: np.ndarray) -> list[np.ndarray]:
    h, w = fg.shape[:2]
    area_threshold = max(96, int(0.0012 * h * w))
    erode_radius = max(1, int(round(0.006 * min(h, w))))
    seed_mask = cv2.erode(fg.astype(np.uint8), _ellipse_kernel(erode_radius), iterations=1) > 0
    num_seed_labels, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(seed_mask.astype(np.uint8), connectivity=8)

    kept_seed_labels = [
        label_idx
        for label_idx in range(1, num_seed_labels)
        if int(seed_stats[label_idx, cv2.CC_STAT_AREA]) >= area_threshold
    ]

    if kept_seed_labels:
        distances = []
        for label_idx in kept_seed_labels:
            seed_component = seed_labels == label_idx
            dist = cv2.distanceTransform((~seed_component).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            distances.append(dist)
        nearest = np.argmin(np.stack(distances, axis=0), axis=0)
        components: list[np.ndarray] = []
        for seed_idx in range(len(kept_seed_labels)):
            component = fg & (nearest == seed_idx)
            component = cv2.morphologyEx(component.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse_kernel(1)) > 0
            if int(component.sum()) >= area_threshold:
                components.append(component)
        if components:
            return components

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < area_threshold:
            continue
        components.append(labels == label_idx)
    return components if components else [fg]


def _component_to_motion_component(rgb: np.ndarray, component_mask: np.ndarray) -> MotionComponent | None:
    ys, xs = np.nonzero(component_mask)
    if xs.size == 0:
        return None

    h, w = rgb.shape[:2]
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    comp_mask = component_mask[y0:y1, x0:x1]
    comp_rgb = rgb[y0:y1, x0:x1]
    ch, cw = comp_mask.shape[:2]
    if ch < 24 or cw < 12:
        return None

    row_left = np.full(ch, np.nan, dtype=np.float32)
    row_right = np.full(ch, np.nan, dtype=np.float32)
    for row_idx in range(ch):
        row_pixels = np.flatnonzero(comp_mask[row_idx])
        if row_pixels.size == 0:
            continue
        row_left[row_idx] = float(row_pixels[0])
        row_right[row_idx] = float(row_pixels[-1])

    valid_rows = np.isfinite(row_left)
    if not valid_rows.any():
        return None

    row_indices = np.arange(ch, dtype=np.float32)
    row_left = np.interp(row_indices, row_indices[valid_rows], row_left[valid_rows]).astype(np.float32)
    row_right = np.interp(row_indices, row_indices[valid_rows], row_right[valid_rows]).astype(np.float32)
    row_center = _smooth_signal((row_left + row_right) * 0.5, sigma=max(1.0, ch / 64.0))
    row_width = np.maximum(1.0, row_right - row_left + 1.0)

    y_rel = row_indices / max(1.0, float(ch - 1))
    upper_rows = (y_rel >= 0.08) & (y_rel <= 0.32)
    if upper_rows.any():
        ref_half_width = float(np.percentile(row_width[upper_rows], 25)) * 0.5
    else:
        ref_half_width = float(np.percentile(row_width, 25)) * 0.5
    ref_half_width = max(8.0, ref_half_width)

    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    x_center = row_center[:, None]
    lateral = np.abs(xx - x_center) / ref_half_width

    edge = _edge_weight(comp_rgb)
    sharp_boundary, soft_boundary = _distance_weights(comp_mask)

    gray = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    upper_gray = gray[y_rel <= 0.65]
    dark_ref = float(np.percentile(upper_gray, 60)) if upper_gray.size else float(np.percentile(gray, 60))
    dark_prior = _soft_clip01((dark_ref - gray + 0.16) / 0.32)

    y_map = yy / max(1.0, float(ch - 1))
    top_focus = np.exp(-(((y_map - 0.16) / 0.16) ** 2)).astype(np.float32)
    mid_focus = np.exp(-(((y_map - 0.47) / 0.20) ** 2)).astype(np.float32)
    lower_focus = _soft_ramp(y_map, 0.50, 0.92)
    lower_tip = _soft_ramp(y_map, 0.72, 1.00)
    side_focus = _soft_ramp(lateral, 0.38, 1.35)
    sleeve_focus = _soft_ramp(lateral, 0.58, 1.55)
    active_support = np.exp(-((lateral / 1.35) ** 4)).astype(np.float32)
    active_support_lower = np.exp(-((lateral / 1.65) ** 4)).astype(np.float32)

    face_center_y = float(0.22 * ch)
    face_center_x = float(row_center[int(min(ch - 1, round(face_center_y)))])
    face_rx = max(9.0, min(ref_half_width * 0.42, cw * 0.28))
    face_ry = max(10.0, ch * 0.14)
    face_block = (((xx - face_center_x) / face_rx) ** 2 + ((yy - face_center_y) / face_ry) ** 2) <= 1.0
    face_block = cv2.GaussianBlur(face_block.astype(np.float32), (0, 0), sigmaX=1.6, sigmaY=1.6)
    face_keepout = 1.0 - _soft_clip01(face_block * 1.8)

    hair = (
        comp_mask.astype(np.float32)
        * sharp_boundary
        * face_keepout
        * active_support
        * (0.55 * top_focus + 0.85 * (top_focus * 0.55 + side_focus * 0.45))
        * (0.30 + 0.70 * np.maximum(edge, dark_prior))
    )
    hair = cv2.GaussianBlur(hair.astype(np.float32), (0, 0), sigmaX=1.6, sigmaY=1.6)
    hair = _soft_clip01(hair * 2.5)

    garment_upper = (
        comp_mask.astype(np.float32)
        * soft_boundary
        * face_keepout
        * active_support
        * mid_focus
        * sleeve_focus
        * (0.32 + 0.68 * edge)
    )
    garment_upper = cv2.GaussianBlur(garment_upper.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
    garment_upper = _soft_clip01(garment_upper * 2.0)

    garment_lower = (
        comp_mask.astype(np.float32)
        * soft_boundary
        * active_support_lower
        * lower_focus
        * (0.55 + 0.45 * np.maximum(side_focus, lower_tip))
        * (0.25 + 0.75 * np.maximum(edge, lower_tip))
    )
    garment_lower = cv2.GaussianBlur(garment_lower.astype(np.float32), (0, 0), sigmaX=2.4, sigmaY=2.4)
    garment_lower = _soft_clip01(garment_lower * 1.8)

    full_face = np.zeros((h, w), dtype=np.float32)
    full_hair = np.zeros((h, w), dtype=np.float32)
    full_upper = np.zeros((h, w), dtype=np.float32)
    full_lower = np.zeros((h, w), dtype=np.float32)
    full_fg = np.zeros((h, w), dtype=np.float32)

    full_face[y0:y1, x0:x1] = _soft_clip01(face_block)
    full_hair[y0:y1, x0:x1] = hair
    full_upper[y0:y1, x0:x1] = garment_upper
    full_lower[y0:y1, x0:x1] = garment_lower
    full_fg[y0:y1, x0:x1] = comp_mask.astype(np.float32)

    return MotionComponent(
        foreground=full_fg,
        face_block=full_face,
        hair=full_hair,
        garment_upper=full_upper,
        garment_lower=full_lower,
    )


def build_motion_parse(rgb: np.ndarray) -> MotionParse:
    fg = _foreground_mask(rgb)
    components = tuple(
        component
        for component in (
            _component_to_motion_component(rgb, component_mask)
            for component_mask in _component_masks(fg)
        )
        if component is not None
    )

    if components:
        face_block = _soft_clip01(np.sum([component.face_block for component in components], axis=0))
        hair = _soft_clip01(np.sum([component.hair for component in components], axis=0))
        garment_upper = _soft_clip01(np.sum([component.garment_upper for component in components], axis=0))
        garment_lower = _soft_clip01(np.sum([component.garment_lower for component in components], axis=0))
    else:
        face_block = np.zeros(fg.shape[:2], dtype=np.float32)
        hair = np.zeros(fg.shape[:2], dtype=np.float32)
        garment_upper = np.zeros(fg.shape[:2], dtype=np.float32)
        garment_lower = np.zeros(fg.shape[:2], dtype=np.float32)

    base_hole = np.maximum.reduce(
        [
            np.clip(hair - 0.08, 0.0, 1.0),
            np.clip(garment_upper - 0.10, 0.0, 1.0),
            np.clip(garment_lower - 0.12, 0.0, 1.0),
        ]
    ).astype(np.float32)
    base_hole = cv2.GaussianBlur(base_hole, (0, 0), sigmaX=1.2, sigmaY=1.2)
    base_hole = _soft_clip01(base_hole)

    return MotionParse(
        foreground=fg.astype(np.float32),
        face_block=face_block,
        hair=hair,
        garment_upper=garment_upper,
        garment_lower=garment_lower,
        base_hole=base_hole,
        components=components,
    )


def build_motion_masks(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    parsed = build_motion_parse(rgb)
    return {
        "hair": parsed.hair,
        "sleeve_left": parsed.garment_upper,
        "sleeve_right": parsed.garment_upper,
        "cloth_edge": parsed.garment_lower,
    }
