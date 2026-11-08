"""Check PNG geometry and pixel alignment of the reusable diagnostic plot."""

from pathlib import Path
from dataclasses import replace
import tempfile
import unittest

from matplotlib.image import imread
import numpy as np

from figure_data_unwrapper import (
    AxisTick, FrameDetection, GridLine, Rectangle, TickGridDetection, TickMark, TickOutlier,
    plot_check,
)


class CheckPlotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "check.png"
        self.image = np.full((60, 80, 3), 255, dtype=np.uint8)
        self.frame = FrameDetection(Rectangle(10, 10, 60, 45), ("left", "bottom"))
        self.ticks = TickGridDetection(
            self.frame,
            (AxisTick(20, (TickMark("bottom", 20, 0, 5),)),), (),
            (GridLine("vertical", 30, 12, 43, 1, 0.5, 1),),
            (GridLine("horizontal", 25, 12, 58, 1, 0.5, 1),),
            None, None, (),
        )

    def source_pixels(self):
        image = np.rint(imread(self.output)[..., :3] * 255).astype(np.uint8)
        return image[:60, :80]

    def test_original_pixels_keep_native_scale_and_position(self):
        rng = np.random.default_rng(6)
        original = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
        path = plot_check(original, output=self.output)
        self.assertEqual(path, self.output)
        np.testing.assert_array_equal(self.source_pixels(), original)
        self.assertEqual(imread(self.output).shape[1], 80 + 280)

    def test_frame_and_grid_are_overlaid_at_image_coordinates(self):
        plot_check(self.image, self.frame, self.ticks, output=self.output)
        pixels = self.source_pixels()
        red = (pixels[..., 0] > 150) & (pixels[..., 1] < 150) & (pixels[..., 2] < 150)
        cyan = (pixels[..., 0] < 150) & (pixels[..., 1] > 100) & (pixels[..., 2] > 100)
        self.assertTrue(red[14:23, 10].all())  # Measured solid left edge.
        inferred = red[14:40, 60]
        self.assertTrue(inferred.any())
        self.assertFalse(inferred.all())  # Inferred right edge is dashed.
        self.assertTrue(cyan[14:23, 30].any())
        self.assertTrue(cyan[25, 33:57].any())
        np.testing.assert_array_equal(pixels[2, 2], [255, 255, 255])

    def test_tick_markers_are_optional(self):
        for enabled in (None, False, True):
            with self.subTest(enabled=enabled):
                options = {} if enabled is None else {"show_ticks": enabled}
                plot_check(self.image, self.frame, self.ticks, output=self.output, **options)
                pixels = self.source_pixels()
                orange = (pixels[..., 0] > 200) & (pixels[..., 1] > 80) & (pixels[..., 1] < 200) & (pixels[..., 2] < 80)
                self.assertEqual(bool(orange.any()), enabled is not False)

    def test_tick_segments_show_measured_direction_and_length_on_all_sides(self):
        ticks = TickGridDetection(
            self.frame,
            (AxisTick(20, (TickMark("top", 20, 3, 8), TickMark("bottom", 20, 3, 8))),),
            (AxisTick(25, (TickMark("left", 25, 3, 8), TickMark("right", 25, 3, 8))),),
            (), (), None, None, (),
        )
        plot_check(self.image, self.frame, ticks, output=self.output)
        pixels = self.source_pixels()
        orange = (pixels[..., 0] > 200) & (pixels[..., 1] > 80) & (pixels[..., 1] < 200) & (pixels[..., 2] < 80)
        # These points lie beyond the root circles and require tick segments.
        for x, y in ((20, 3), (20, 52), (3, 25), (67, 25)):
            with self.subTest(x=x, y=y):
                self.assertTrue(orange[y, x])

    def test_frame_can_be_taken_from_tick_analysis(self):
        plot_check(self.image, ticks=self.ticks, output=self.output)
        pixels = self.source_pixels()
        self.assertLess(int(pixels[16, 10, 1]), 150)

    def test_separated_ticks_remain_visible_in_purple(self):
        rejected = TickOutlier(AxisTick(45, (TickMark("bottom", 45, 0, 8),)), "off_reference", 10)
        ticks = replace(self.ticks, rejected_x_ticks=(rejected,))
        plot_check(self.image, ticks=ticks, output=self.output)
        pixels = self.source_pixels()
        purple = (pixels[..., 0] > 120) & (pixels[..., 1] < 100) & (pixels[..., 2] > 150)
        self.assertTrue(purple[42:55, 42:49].any())

    def test_colors_add_swatches_without_covering_source(self):
        plot_check(self.image, colors=[(255, 255, 255, 20), (31, 119, 180, 10)], output=self.output)
        image = np.rint(imread(self.output)[..., :3] * 255).astype(np.uint8)
        np.testing.assert_array_equal(self.source_pixels(), self.image)
        np.testing.assert_array_equal(image[242, 80 + 26], [31, 119, 180])

    def test_failure_still_saves_original_image(self):
        plot_check(self.image, output=self.output)
        np.testing.assert_array_equal(self.source_pixels(), self.image)

    def test_grayscale_and_transparency_render_as_visible_rgb(self):
        for source, expected in (
            (np.full((60, 80), 128, dtype=np.uint8), 128),
            (np.zeros((60, 80, 4)), 255),
            (np.zeros((60, 80), dtype=bool), 255),
        ):
            with self.subTest(shape=source.shape, dtype=source.dtype):
                plot_check(source, output=self.output)
                np.testing.assert_array_equal(self.source_pixels(), np.full((60, 80, 3), expected))

    def test_mismatched_analysis_is_rejected(self):
        other = FrameDetection(Rectangle(10, 10, 55, 45), ("left", "bottom"))
        with self.assertRaises(ValueError):
            plot_check(self.image, other, self.ticks, output=self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_output_bounds_and_colors_are_rejected(self):
        with self.assertRaises(ValueError):
            plot_check(self.image, output=self.output.with_suffix(".jpg"))
        outside = FrameDetection(Rectangle(10, 10, 100, 45), ("left", "bottom"))
        with self.assertRaises(ValueError):
            plot_check(self.image, outside, output=self.output)
        with self.assertRaises(ValueError):
            plot_check(self.image, colors=[(255, 0, 0, -1)], output=self.output)
        self.assertFalse(self.output.exists())
