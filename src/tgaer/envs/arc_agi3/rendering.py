from __future__ import annotations

import base64
import io

# ARC colour palette: grid index 0-15 -> RGB. Index 0 is the background/black.
ARC_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
    (60, 60, 60),
    (0, 80, 160),
    (180, 40, 30),
    (30, 140, 40),
    (180, 150, 0),
    (90, 90, 90),
]


def grid_to_rgb(frame: list[list[list[int]]] | None):
    """Render the most recent grid in an ARC-AGI-3 frame to an HxWx3 uint8 RGB
    array via the palette. Returns None if the frame is empty/malformed."""
    import numpy as np

    if not frame:
        return None
    arr = np.asarray(frame[-1], dtype=int)
    if arr.ndim != 2:
        return None
    palette = np.asarray(ARC_PALETTE, dtype=np.uint8)
    return palette[np.clip(arr, 0, len(palette) - 1)]


def grid_to_png_data_url(frame: list[list[list[int]]] | None, scale: int = 8) -> str | None:
    """Render the frame to a PNG (nearest-neighbour upscaled by ``scale`` so each
    cell is a visible block) and return it as a base64 ``data:`` URL for the
    OpenAI vision message format. Returns None if the frame is empty."""
    from PIL import Image

    rgb = grid_to_rgb(frame)
    if rgb is None:
        return None
    img = Image.fromarray(rgb, "RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
