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


def build_motion_masks(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = rgb.shape[:2]
    ew = _edge_weight(rgb)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx / max(1.0, w - 1.0)) * 2.0 - 1.0
    yn = (yy / max(1.0, h - 1.0)) * 2.0 - 1.0

    # Hair prior: upper-middle ellipse, with edges.
    hair_prior = np.exp(-((xn / 0.55) ** 2 + ((yn + 0.55) / 0.35) ** 2)).astype(np.float32)
    hair = _soft_clip01(hair_prior * (0.35 + 0.95 * ew))
    hair = cv2.GaussianBlur(hair, (0, 0), sigmaX=6.0, sigmaY=6.0)

    # Sleeves prior: mid-lower left/right bands, with edges.
    left_band = np.exp(-(((xn + 0.65) / 0.28) ** 2 + ((yn - 0.10) / 0.55) ** 2)).astype(np.float32)
    right_band = np.exp(-(((xn - 0.65) / 0.28) ** 2 + ((yn - 0.10) / 0.55) ** 2)).astype(np.float32)
    sleeves_prior = np.maximum(left_band, right_band)
    sleeves = _soft_clip01(sleeves_prior * (0.25 + 1.05 * ew))
    sleeves = cv2.GaussianBlur(sleeves, (0, 0), sigmaX=7.0, sigmaY=7.0)

    # Cloth edge: lower/middle area with edges; helps capes/robes.
    cloth_prior = np.exp(-((xn / 0.95) ** 2 + ((yn - 0.15) / 0.75) ** 2)).astype(np.float32)
    cloth_edge = _soft_clip01(cloth_prior * ew)
    cloth_edge = cv2.GaussianBlur(cloth_edge, (0, 0), sigmaX=8.0, sigmaY=8.0)

    return {"hair": hair, "sleeves": sleeves, "cloth_edge": cloth_edge}

