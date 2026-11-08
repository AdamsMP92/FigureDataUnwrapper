"""Tick/grid checks against independent Matplotlib pixel geometry."""

import unittest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from figure_data_unwrapper import detect_ticks_and_grid


def render_case(*, direction="out", grid=None, partial=False, opposite=False,
                endpoints=False, ticks=True, minor=False, logarithmic=False,
                tick_color="black", background="white"):
    figure = Figure(figsize=(4.5, 3.5), dpi=100, facecolor=background)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.20, 0.20, 0.70, 0.70), facecolor=background)
    x = [0, 5, 10] if endpoints else [1, 3, 6, 9]
    y = [0, 5, 10] if endpoints else [2, 5, 8]
    axes.set(xlim=(0, 10), ylim=(0, 10))
    if logarithmic:
        axes.set_xscale("log")
        axes.set_xlim(1, 1000)
        x = [2, 5, 10, 50, 100, 500]
    axes.minorticks_off()
    axes.set_xticks(x)
    axes.set_yticks(y)
    axes.tick_params(which="both", direction=direction, length=5,
                     colors=tick_color, labelsize=7,
                     bottom=ticks, left=ticks, top=ticks and opposite, right=ticks and opposite)
    if partial:
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
    for spine in axes.spines.values():
        spine.set_color(tick_color)
    if minor:
        x = [1, 5, 9]
        axes.set_xticks(x)
        axes.set_xticks([2, 3, 4, 6, 7, 8], minor=True)
        axes.tick_params(which="minor", length=2.5)
    if grid:
        axes.grid(color="#aaaaaa", linewidth=0.7, linestyle=grid)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3].copy() / 255
    x_pixels = axes.transData.transform(np.column_stack((x, np.zeros(len(x)))))[:, 0] - 0.5
    y_pixels = image.shape[0] - axes.transData.transform(
        np.column_stack((np.full(len(y), x[0]), y))
    )[:, 1] - 0.5
    return image, np.sort(x_pixels), np.sort(y_pixels)


class RenderedTickGridTests(unittest.TestCase):
    def assert_positions(self, actual, expected):
        np.testing.assert_allclose([item.position for item in actual], expected, atol=1, rtol=0)

    def test_tick_directions_and_grid_styles(self):
        for direction in ("in", "out", "inout"):
            for grid in (None, "-", "--", ":"):
                with self.subTest(direction=direction, grid=grid):
                    image, x, y = render_case(direction=direction, grid=grid)
                    result = detect_ticks_and_grid(image)
                    self.assertIsNotNone(result)
                    self.assert_positions(result.x_ticks, x)
                    self.assert_positions(result.y_ticks, y)
                    self.assert_positions(result.vertical_grid, x if grid else [])
                    self.assert_positions(result.horizontal_grid, y if grid else [])

    def test_partial_frame_and_opposing_ticks(self):
        for partial, opposite in ((True, False), (False, True)):
            with self.subTest(partial=partial, opposite=opposite):
                image, x, y = render_case(partial=partial, opposite=opposite, grid="--")
                result = detect_ticks_and_grid(image)
                self.assertIsNotNone(result)
                self.assert_positions(result.x_ticks, x)
                self.assert_positions(result.y_ticks, y)
                for tick in result.x_ticks:
                    self.assertEqual(set(tick.sides), {"bottom", "top"} if opposite else {"bottom"})

    def test_grid_without_tick_marks(self):
        image, x, y = render_case(ticks=False, grid=":")
        result = detect_ticks_and_grid(image)
        self.assertIsNotNone(result)
        self.assertEqual((result.x_ticks, result.y_ticks), ((), ()))
        self.assert_positions(result.vertical_grid, x)
        self.assert_positions(result.horizontal_grid, y)

    def test_endpoint_ticks_in_automatic_pipeline(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                image, x, y = render_case(endpoints=True, partial=partial)
                result = detect_ticks_and_grid(image)
                self.assertIsNotNone(result)
                self.assert_positions(result.x_ticks, x)
                self.assert_positions(result.y_ticks, y)
                self.assertEqual((result.vertical_grid, result.horizontal_grid), ((), ()))

    def test_color_and_background_variants(self):
        for color, background in (("yellow", "white"), ("cyan", "white"),
                                  ("#cccccc", "white"), ("white", "black")):
            with self.subTest(color=color, background=background):
                image, x, y = render_case(tick_color=color, background=background)
                result = detect_ticks_and_grid(image)
                self.assertIsNotNone(result)
                self.assert_positions(result.x_ticks, x)
                self.assert_positions(result.y_ticks, y)

    def test_logarithmic_spacing_is_not_forced_to_linear_spacing(self):
        image, x, y = render_case(logarithmic=True)
        result = detect_ticks_and_grid(image)
        self.assertIsNotNone(result)
        self.assert_positions(result.x_ticks, x)
        self.assert_positions(result.y_ticks, y)

    def test_optional_major_tick_count(self):
        image, x, y = render_case(minor=True)
        automatic = detect_ticks_and_grid(image)
        self.assertIsNotNone(automatic)
        self.assertEqual(len(automatic.x_ticks), 9)
        guided = detect_ticks_and_grid(image, expected_x_ticks=3, expected_y_ticks=3)
        self.assert_positions(guided.x_ticks, x)
        self.assert_positions(guided.y_ticks, y)
        self.assertTrue(guided.x_count_matches and guided.y_count_matches)
