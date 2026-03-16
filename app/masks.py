from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def _soft_clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _edge_weight(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return _soft_clip01(mag)


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
    if fg.mean() < 0.05 or fg.mean() > 0.95:
        return None
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    return fg


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    fg = _grabcut_foreground(rgb)
    if fg is None:
        fg = ~_background_mask(rgb)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    return fg


def _face_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return np.zeros(gray.shape[:2], dtype=bool)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    mask = np.zeros(gray.shape[:2], dtype=bool)
    for (x, y, w, h) in faces:
        pad_x = int(w * 0.18)
        pad_y = int(h * 0.22)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(gray.shape[1], x + w + pad_x)
        y1 = min(gray.shape[0], y + h + int(h * 0.10))
        mask[y0:y1, x0:x1] = True
    if not mask.any():
        h, w = gray.shape[:2]
        cx = int(w * 0.5)
        cy = int(h * 0.34)
        rx = int(w * 0.14)
        ry = int(h * 0.18)
        y, x = np.ogrid[:h, :w]
        ellipse = (((x - cx) / max(1, rx)) ** 2 + ((y - cy) / max(1, ry)) ** 2) <= 1.0
        mask |= ellipse
    return mask


def _boundary_weights(fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = fg.shape[:2]
    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    sharp_scale = max(3.0, 0.045 * min(h, w))
    soft_scale = max(10.0, 0.20 * min(h, w))
    sharp = np.exp(-((dist / sharp_scale) ** 2)).astype(np.float32)
    soft = np.exp(-((dist / soft_scale) ** 2)).astype(np.float32)
    return _soft_clip01(sharp), _soft_clip01(soft)


def build_motion_masks(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = rgb.shape[:2]
    ew = _edge_weight(rgb)
    fg = _foreground_mask(rgb)
    bw_sharp, bw_soft = _boundary_weights(fg)
    face = _face_mask(rgb)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx / max(1.0, w - 1.0)) * 2.0 - 1.0
    yn = (yy / max(1.0, h - 1.0)) * 2.0 - 1.0
    fg_f = fg.astype(np.float32)

    if face.any():
        face_dist = cv2.distanceTransform((~face).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        d0 = 0.05 * min(h, w)
        d1 = max(8.0, 0.16 * min(h, w))
        face_far = _soft_clip01((face_dist - d0) / d1)
        face_block = cv2.dilate(face.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1) > 0
    else:
        face_far = np.ones((h, w), dtype=np.float32)
        face_block = np.zeros((h, w), dtype=bool)

    # Hair should mostly move on outer contour and tips, not near face/root.
    crown = np.exp(-((xn / 0.62) ** 2 + ((yn + 0.58) / 0.42) ** 2)).astype(np.float32)
    long_hair = np.exp(-((xn / 0.52) ** 2 + ((yn + 0.03) / 0.95) ** 2)).astype(np.float32)
    hair_prior = np.maximum(crown, 0.75 * long_hair)
    hair = _soft_clip01(hair_prior * (0.30 + 1.10 * ew) * bw_sharp * face_far * fg_f)
    hair = cv2.GaussianBlur(hair, (0, 0), sigmaX=6.0, sigmaY=6.0)

    # Sleeves/arms: split left/right so each side can sway coherently.
    left_band = np.exp(-(((xn + 0.62) / 0.34) ** 2 + ((yn - 0.18) / 0.72) ** 2)).astype(np.float32)
    right_band = np.exp(-(((xn - 0.62) / 0.34) ** 2 + ((yn - 0.18) / 0.72) ** 2)).astype(np.float32)
    sleeve_mix = (0.45 * bw_sharp + 0.55 * bw_soft) * fg_f
    sleeve_left = _soft_clip01(left_band * (0.30 + 1.25 * ew) * sleeve_mix)
    sleeve_right = _soft_clip01(right_band * (0.30 + 1.25 * ew) * sleeve_mix)
    sleeve_left = cv2.GaussianBlur(sleeve_left, (0, 0), sigmaX=7.0, sigmaY=7.0)
    sleeve_right = cv2.GaussianBlur(sleeve_right, (0, 0), sigmaX=7.0, sigmaY=7.0)

    # Cloth flow: emphasize lower cloth area, not just tiny contour fragments.
    cloth_prior = np.exp(-((xn / 1.05) ** 2 + ((yn - 0.34) / 0.95) ** 2)).astype(np.float32)
    cloth_edge = _soft_clip01(cloth_prior * (0.24 + 1.05 * ew) * (0.30 + 0.70 * bw_soft) * fg_f)
    cloth_edge = cv2.GaussianBlur(cloth_edge, (0, 0), sigmaX=8.0, sigmaY=8.0)

    if face_block.any():
        mask_keep = (~face_block).astype(np.float32)
        hair *= mask_keep
        sleeve_left *= mask_keep
        sleeve_right *= mask_keep
        cloth_edge *= mask_keep

    return {
        "hair": hair,
        "sleeve_left": sleeve_left,
        "sleeve_right": sleeve_right,
        "cloth_edge": cloth_edge,
    }
