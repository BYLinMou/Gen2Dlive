from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


def _sample_palette(rgb: np.ndarray, rng: np.random.Generator, k: int = 32) -> List[Tuple[int, int, int]]:
    h, w = rgb.shape[:2]
    ys = rng.integers(0, h, size=k)
    xs = rng.integers(0, w, size=k)
    colors = []
    for y, x in zip(ys, xs):
        r, g, b = rgb[int(y), int(x)]
        colors.append((int(b), int(g), int(r)))  # BGR
    return colors


@dataclass(frozen=True)
class _Particle:
    cx: float
    cy: float
    ax: float
    ay: float
    phase: float
    radius: float
    color_bgr: Tuple[int, int, int]
    alpha: float


class ParticleField:
    def __init__(self, particles: List[_Particle]) -> None:
        self._particles = particles

    @property
    def count(self) -> int:
        return len(self._particles)

    @staticmethod
    def from_image(rgb: np.ndarray, *, count: int, seed: int) -> "ParticleField":
        if count <= 0:
            return ParticleField([])

        rng = np.random.default_rng(seed)
        palette = _sample_palette(rgb, rng, k=64)
        h, w = rgb.shape[:2]

        particles: List[_Particle] = []
        for _ in range(count):
            cx = float(rng.uniform(0.05, 0.95) * w)
            cy = float(rng.uniform(0.05, 0.95) * h)
            ax = float(rng.uniform(6.0, 26.0))
            ay = float(rng.uniform(8.0, 30.0))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            radius = float(rng.uniform(1.5, 3.8))
            color = palette[int(rng.integers(0, len(palette)))]
            alpha = float(rng.uniform(0.14, 0.35))
            particles.append(
                _Particle(
                    cx=cx,
                    cy=cy,
                    ax=ax,
                    ay=ay,
                    phase=phase,
                    radius=radius,
                    color_bgr=color,
                    alpha=alpha,
                )
            )
        return ParticleField(particles)

    def render_frame_bgr(self, w: int, h: int, *, t: int, total_frames: int) -> np.ndarray:
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        if not self._particles:
            return overlay

        phase_t = 2.0 * math.pi * (t / total_frames)
        for p in self._particles:
            x = p.cx + p.ax * math.sin(phase_t + p.phase)
            y = p.cy + p.ay * math.cos(phase_t + p.phase * 0.7)
            a = p.alpha * (0.65 + 0.35 * math.sin(phase_t + p.phase + 1.2))
            if a <= 0.01:
                continue
            if x < -10 or x > w + 10 or y < -10 or y > h + 10:
                continue

            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.circle(tmp, (int(x), int(y)), int(p.radius), p.color_bgr, thickness=-1, lineType=cv2.LINE_AA)
            tmp = cv2.GaussianBlur(tmp, (0, 0), sigmaX=1.2, sigmaY=1.2)
            overlay = cv2.addWeighted(overlay, 1.0, tmp, float(a), 0.0)
        return overlay

