from __future__ import annotations

import colorsys
from typing import List, Tuple


def get_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate visually distinct RGB colors using the golden-ratio hue step."""
    colors = []
    golden_ratio = 0.618033988749895
    for i in range(max(n, 0)):
        hue = (i * golden_ratio) % 1.0
        saturation = 0.75 + (i % 3) * 0.08
        value = 0.85 - (i % 2) * 0.1
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors
