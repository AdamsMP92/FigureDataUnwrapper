"""Foreground/background binarization for figures with a uniform background."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _normalized_image(image: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(image, dtype=np.float64)
    if (
        values.ndim not in (2, 3)
        or values.size == 0
        or (values.ndim == 3 and values.shape[2] not in (3, 4))
    ):
        raise ValueError("Expected a nonempty grayscale, RGB, or RGBA image.")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Image intensities must be finite and in [0, 1].")
    if values.ndim == 3 and values.shape[2] == 4:
        # Interpret transparency against a white page.
        alpha = values[..., 3:4]
        values = values[..., :3] * alpha + (1 - alpha)
    return values


def foreground_mask(
    image: ArrayLike, *, min_contrast: float = 0.03
) -> NDArray[np.bool_]:
    """Return True for pixels differing from the estimated page background.

    Input intensities are normalized to [0, 1]. The median image perimeter
    estimates a uniform background. Grayscale uses absolute intensity contrast;
    RGB uses the largest absolute channel difference, retaining e.g. yellow
    strokes without first reducing their color contrast to luminance.
    RGBA is composited over white. Boolean input already denotes foreground.

    min_contrast is an absolute normalized difference, not a probability.
    Margins should predominantly show the page background. Nonuniform lighting
    or a frame occupying the entire perimeter needs other preprocessing.
    """
    if not 0 < min_contrast < 1:
        raise ValueError("min_contrast must be between 0 and 1.")
    source = np.asarray(image)
    if source.dtype == np.bool_ and source.ndim == 2 and source.size:
        return source.copy()
    return _foreground_contrast(source) >= min_contrast


def _foreground_contrast(image: ArrayLike) -> NDArray[np.float64]:
    """Absolute background contrast, retaining strength for branch analysis."""
    source = np.asarray(image)
    if source.dtype == np.bool_ and source.ndim == 2 and source.size:
        return source.astype(np.float64)
    values = _normalized_image(source)
    perimeter = np.concatenate((
        values[0], values[-1], values[1:-1, 0], values[1:-1, -1],
    ), axis=0)
    background = np.median(perimeter, axis=0)
    contrast = np.abs(values - background)
    if values.ndim == 3:
        contrast = contrast.max(axis=2)
    return contrast
