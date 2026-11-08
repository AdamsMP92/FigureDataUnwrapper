"""An axis-aligned rectangle model in image pixel coordinates."""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Rectangle:
    """Four edge centerlines; x increases rightward and y downward.

    Integer coordinates refer to pixel centers. The bounds describe the
    geometry of the outline, rather than a half-open array slice.
    """

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        bounds = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("Rectangle coordinates must be finite.")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("A rectangle must have positive width and height.")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


def rectangle_model(
    shape: tuple[int, int], rectangle: Rectangle, *, line_width: float = 1.0
) -> NDArray[np.bool_]:
    """Rasterize a rectangular outline as a boolean mask of shape (height, width).

    The stroke is centered on the four edges, with square corners. Pixels
    outside the image are clipped. True marks the outline, not its interior.
    """
    if len(shape) != 2 or any(
        not isinstance(size, (int, np.integer)) or size <= 0 for size in shape
    ):
        raise ValueError("shape must contain two positive integer dimensions.")
    if not math.isfinite(line_width) or line_width <= 0:
        raise ValueError("line_width must be finite and positive.")

    y, x = np.ogrid[:shape[0], :shape[1]]
    # Signed distance in the max norm: negative inside, zero on the border.
    distance = np.maximum(
        np.maximum(rectangle.left - x, x - rectangle.right),
        np.maximum(rectangle.top - y, y - rectangle.bottom),
    )
    return np.abs(distance) <= line_width / 2
