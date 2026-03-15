from __future__ import annotations

from PIL import Image


def resize_to_square_cover(img: Image.Image, *, size: int) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image size")
    scale = max(size / w, size / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return resized.crop((left, top, left + size, top + size))

