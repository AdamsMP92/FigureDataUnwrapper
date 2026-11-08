"""Replace selected color families with a uniform page background."""

from dataclasses import dataclass

from matplotlib.colors import rgb_to_hsv
import numpy as np
from numpy.typing import ArrayLike

from .extraction import _color_rgb, _positive_number
from .preprocessing import _normalized_image


@dataclass(frozen=True)
class ColorRemoval:
    """Normalized RGB copy and a True-for-removed mask; the source is untouched.

    The mask identifies discarded/unknown image content. Background replacement
    does not establish what a covered curve looked like at these positions.
    Per-target counts can overlap; removed_pixels counts the union once.
    """

    image: np.ndarray
    mask: np.ndarray
    background_color: tuple[float, float, float]
    target_colors: tuple[tuple[float, float, float], ...]
    counts: tuple[int, ...]
    mode: str

    @property
    def removed_pixels(self) -> int:
        return int(np.count_nonzero(self.mask))

    def __iter__(self):
        yield self.image
        yield self.mask


def _targets(colors):
    if isinstance(colors, str):
        return [_color_rgb(colors)]
    values = np.asarray(colors) if isinstance(colors, np.ndarray) else None
    if values is not None and values.shape == (3,) and np.issubdtype(values.dtype, np.number):
        return [_color_rgb(values)]
    try:
        sequence = tuple(colors)
    except TypeError as error:
        raise ValueError("colors must contain color names, hex strings or RGB triples.") from error
    if len(sequence) == 3 and all(isinstance(v, (int, float, np.number)) for v in sequence):
        return [_color_rgb(sequence)]
    return [_color_rgb(value) for value in sequence]


def remove_colors(
    image: ArrayLike, colors, *, background_color=None, mode: str = "hue",
    hue_tolerance: float = 15., min_saturation: float = .1,
    rgb_tolerance: float = .15, min_contrast: float = .03,
) -> ColorRemoval:
    """Remove approximate colors, including antialiasing shades, from a copy.

    Hue mode matches circular HSV hue within hue_tolerance degrees, independently
    of brightness, for pixels with saturation >= min_saturation. It preserves
    neutral grayscale pixels, including black/gray curves and white backgrounds.
    Only chromatic target colors are valid in hue mode. For black/gray/white use
    mode='rgb', with Euclidean distance rgb_tolerance in normalized RGB.

    Infer a uniform background from the median image perimeter unless explicitly
    supplied. min_contrast protects pixels already close to that background.
    Accept float RGB/RGBA/grayscale [0,1] or integer [0,255]. RGBA is composited
    over white, grayscale expanded to RGB; output is always float RGB [0,1].
    colors accepts one color name/hex/RGB triple, or a sequence of colors.
    Empty color sequences return an unchanged RGB copy and an empty mask.
    """
    if mode not in ("hue", "rgb"):
        raise ValueError("mode must be 'hue' or 'rgb'.")
    hue_tolerance = _positive_number(hue_tolerance, "hue_tolerance", allow_zero=True)
    min_saturation = _positive_number(min_saturation, "min_saturation")
    rgb_tolerance = _positive_number(rgb_tolerance, "rgb_tolerance", allow_zero=True)
    min_contrast = _positive_number(min_contrast, "min_contrast", allow_zero=True)
    if hue_tolerance > 180 or min_saturation > 1 or rgb_tolerance > np.sqrt(3) or min_contrast > np.sqrt(3):
        raise ValueError("Invalid hue, saturation, RGB or contrast tolerance.")
    targets = _targets(colors)
    source = np.asarray(image)
    if np.issubdtype(source.dtype, np.integer):
        if not source.size or source.min() < 0 or source.max() > 255:
            raise ValueError("Integer image channels must be in [0, 255].")
        source = source.astype(float) / 255
    rgb = _normalized_image(source)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if background_color is None:
        perimeter = np.concatenate((rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]))
        background = np.median(perimeter, axis=0)
    else:
        background = _color_rgb(background_color)
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    foreground = np.linalg.norm(rgb - background, axis=2) > min_contrast
    hsv = rgb_to_hsv(rgb) if mode == "hue" and targets else None
    counts = []
    for target in targets:
        if mode == "hue":
            target_hsv = rgb_to_hsv(target)
            if target_hsv[1] <= 1e-8:
                raise ValueError("Achromatic targets need mode='rgb'; their hue is undefined.")
            distance = np.abs(hsv[..., 0] - target_hsv[0])
            selected = ((np.minimum(distance, 1 - distance) * 360 <= hue_tolerance + 1e-10)
                        & (hsv[..., 1] >= min_saturation - 1e-12) & foreground)
        else:
            selected = (np.linalg.norm(rgb - target, axis=2) <= rgb_tolerance + 1e-12) & foreground
        counts.append(int(np.count_nonzero(selected)))
        mask |= selected
    output = rgb.copy()
    output[mask] = background
    return ColorRemoval(output, mask, tuple(float(v) for v in background),
                        tuple(tuple(float(v) for v in target) for target in targets), tuple(counts), mode)
