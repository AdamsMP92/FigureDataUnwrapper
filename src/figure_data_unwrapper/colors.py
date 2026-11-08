"""Exact color-frequency rankings as supplementary figure information."""

import numpy as np
from numpy.typing import ArrayLike

from .preprocessing import _normalized_image


ColorCount = tuple[int | float, int | float, int | float, int]


def rank_colors(image: ArrayLike, n_colors: int) -> list[ColorCount]:
    """Return up to n_colors tuples (R, G, B, pixel_count), most frequent first.

    Accept RGB, RGBA, or 2D grayscale images. Integer channels are 8-bit values
    in [0, 255]; floating channels are normalized to [0, 1]. Output keeps that
    scale. Grayscale is replicated into RGB. RGBA is composited over white;
    integer RGBA results are rounded to 8-bit RGB after compositing.

    Count exact RGB values without clustering or quantizing float colors.
    Antialiasing shades therefore remain distinct. Ties are ordered by ascending
    (R, G, B). Zero returns an empty list; asking for more than the number of
    distinct colors returns all colors. Counts refer to the entire input image.
    """
    if isinstance(n_colors, (bool, np.bool_)) or not isinstance(n_colors, (int, np.integer)):
        raise ValueError("n_colors must be a nonnegative integer.")
    if n_colors < 0:
        raise ValueError("n_colors must be a nonnegative integer.")
    source = np.asarray(image)
    if np.issubdtype(source.dtype, np.integer):
        if source.size == 0 or source.min() < 0 or source.max() > 255:
            raise ValueError("Integer image channels must be in [0, 255].")
        # Reuse the shape/alpha convention of the extraction pipeline.
        rgb = _normalized_image(source.astype(np.float64) / 255)
        rgb = np.rint(rgb * 255).astype(np.uint8)
    elif np.issubdtype(source.dtype, np.floating):
        rgb = _normalized_image(source)
    else:
        raise ValueError("Expected integer or floating image channels.")
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if n_colors == 0:
        return []

    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    # np.unique orders RGB lexicographically; stable ranking preserves this
    # deterministic order when different colors occupy the same pixel count.
    order = np.argsort(-counts, kind="stable")[:n_colors]
    return [(*colors[index].tolist(), int(counts[index])) for index in order]
