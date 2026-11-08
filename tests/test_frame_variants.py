"""Extraction challenges for colors and all combinations of visible sides.

All color and visibility targets are retained from the original challenge
matrix. No skips or expectedFailure markers hide unsupported cases.
"""

from dataclasses import dataclass
from itertools import combinations
import unittest

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from figure_data_unwrapper import Rectangle, detect_frame


COLORS = {
    "black": "#000000",
    "blue": "#0000ff",
    "red": "#ff0000",
    "dark_green": "#008000",
    "purple": "#800080",
    "orange": "#ff8000",
    "cyan": "#00ffff",
    "light_gray": "#bfbfbf",
    "yellow": "#ffff00",
}
SIDES = ("left", "bottom", "right", "top")
SIDE_COMBINATIONS = tuple(
    sides for count in range(5) for sides in combinations(SIDES, count)
)


@dataclass(frozen=True)
class FrameCase:
    color: str
    sides: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.color}__{'_'.join(self.sides) or 'none'}"

    @property
    def expects_frame(self) -> bool:
        # Each finite spine spans the full plot edge. Any two of them specify
        # all four bounds via their endpoints; a single side leaves one
        # dimension undetermined. This is a fixture assumption, not a claim
        # that every cropped or occluded scientific figure is identifiable.
        return len(self.sides) >= 2


CASES = tuple(
    FrameCase(color, sides) for color in COLORS for sides in SIDE_COMBINATIONS
)


def render_case(case: FrameCase) -> tuple[np.ndarray, Rectangle]:
    """Render independently with Matplotlib; retain bounds only as test truth."""
    figure = Figure(figsize=(3.2, 2.4), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.18, 0.20, 0.74, 0.70))
    axes.scatter([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4], s=12, color="#333333")
    axes.set(xlim=(-2.5, 2.5), ylim=(-0.5, 4.5))
    axes.set_xticks([-2, 0, 2])
    axes.set_yticks([0, 2, 4])
    # Negative controls have no grid that could independently reveal bounds.
    if case.expects_frame:
        axes.grid(color="#dddddd", linewidth=0.6)
    for side, spine in axes.spines.items():
        spine.set_visible(side in case.sides)
        spine.set_color(COLORS[case.color])
        spine.set_linewidth(1.2)
    axes.tick_params(
        left="left" in case.sides,
        bottom="bottom" in case.sides,
        right=False,
        top=False,
        labelsize=7,
    )
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy() / 255
    bounds = axes.get_window_extent()
    height = rgb.shape[0]
    expected = Rectangle(
        bounds.x0 - 0.5,
        height - bounds.y1 - 0.5,
        bounds.x1 - 0.5,
        height - bounds.y0 - 0.5,
    )
    return rgb, expected


class FrameVariantTests(unittest.TestCase):
    """Each generated test checks one color/visibility combination."""


def _make_test(case: FrameCase):
    def test(self):
        self.case = case
        self.detected = None
        self.max_error = None
        rgb, expected = render_case(case)
        grayscale = rgb @ np.array([0.2126, 0.7152, 0.0722])
        # The detector receives only pixels and its unchanged default options.
        self.detected = detect_frame(grayscale)
        if not case.expects_frame:
            self.assertIsNone(self.detected, "Insufficient sides: expected abstention.")
            return
        self.assertIsNotNone(
            self.detected,
            f"No frame extracted: color={case.color}, visible sides={case.sides}.",
        )
        actual_bounds = np.array([
            self.detected.left, self.detected.top,
            self.detected.right, self.detected.bottom,
        ])
        expected_bounds = np.array([
            expected.left, expected.top, expected.right, expected.bottom,
        ])
        self.max_error = float(np.max(np.abs(actual_bounds - expected_bounds)))
        self.assertLessEqual(self.max_error, 1.0, "Frame bounds differ by more than 1 pixel.")

    test.__doc__ = (
        f"{case.color}: {', '.join(case.sides) or 'no sides'}; "
        f"expect {'reconstruction' if case.expects_frame else 'abstention'}."
    )
    return test


for _case in CASES:
    setattr(FrameVariantTests, f"test_{_case.name}", _make_test(_case))


if __name__ == "__main__":
    unittest.main()
