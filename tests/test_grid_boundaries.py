"""Regressions for frame-rooted grid tracing rather than interior projections."""

from pathlib import Path
import unittest

from matplotlib.image import imread
import numpy as np

from figure_data_unwrapper import detect_ticks_and_grid
from test_ticks import FRAME, X, Y, fixture


class GridBoundaryTests(unittest.TestCase):
    def assert_positions(self, lines, expected):
        np.testing.assert_allclose([line.position for line in lines], expected, atol=1, rtol=0)

    def assert_no_grid(self, image, **kwargs):
        result = detect_ticks_and_grid(image, frame=FRAME, **kwargs)
        self.assertEqual(result.vertical_grid, ())
        self.assertEqual(result.horizontal_grid, ())

    def test_real_paper_legend_is_not_a_grid(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "real_example_1" / "real_example1.png"
        result = detect_ticks_and_grid(imread(path))
        self.assertIsNotNone(result)
        self.assertEqual(result.vertical_grid, ())
        self.assertEqual(result.horizontal_grid, ())

    def test_legend_and_text_fragments_do_not_replace_real_grid(self):
        image = fixture(ticks=False, grid=True)
        # Large foreground fractions and nearly full spans, but no attachment.
        for start, end in ((41, 71), (99, 125), (151, 181)):
            image[65, start:end] = 0
        for start, end in ((27, 50), (65, 91), (103, 125)):
            image[start:end, 110] = 0
        result = detect_ticks_and_grid(image, frame=FRAME)
        self.assert_positions(result.vertical_grid, X)
        self.assert_positions(result.horizontal_grid, Y)

    def test_boundary_stubs_cannot_bridge_detached_interior_lines(self):
        image = fixture(ticks=False)
        for start, end in ((31, 37), (45, 180), (184, 190)):
            image[65, start:end] = 0.8
        for start, end in ((21, 27), (35, 120), (124, 130)):
            image[start:end, 110] = 0.8
        self.assert_no_grid(image)

    def test_full_span_grid_must_reach_both_boundaries(self):
        image = fixture(ticks=False)
        image[65, 31:179] = 0.8
        image[21:119, 110] = 0.8
        self.assert_no_grid(image)

    def test_small_internal_gaps_are_allowed(self):
        image = fixture(ticks=False, grid=True)
        image[75, 100:103] = 1
        image[60:64, 90] = 1
        result = detect_ticks_and_grid(image, frame=FRAME)
        self.assert_positions(result.vertical_grid, X)
        self.assert_positions(result.horizontal_grid, Y)
        for line in result.vertical_grid:
            self.assertEqual(line.frame_sides, ("top", "bottom"))
        for line in result.horizontal_grid:
            self.assertEqual(line.frame_sides, ("left", "right"))

    def test_gap_tolerance_is_explicit_and_does_not_fill_arbitrary_spaces(self):
        image = fixture(ticks=False)
        image[65, 31:190] = 0.8
        image[65, 80:86] = 1
        self.assert_no_grid(image)
        result = detect_ticks_and_grid(image, frame=FRAME, grid_max_gap=6)
        self.assert_positions(result.horizontal_grid, [65])

    def test_thick_frame_and_dark_background_keep_grid_positions(self):
        image = fixture(ticks=False, grid=True)
        image[18:23, 28:193] = image[128:133, 28:193] = 0
        image[18:133, 28:33] = image[18:133, 188:193] = 0
        for source in (image, 1 - image):
            with self.subTest(dark_background=source is not image):
                result = detect_ticks_and_grid(source, frame=FRAME)
                self.assert_positions(result.vertical_grid, X)
                self.assert_positions(result.horizontal_grid, Y)

    def test_inferred_frame_boundaries_also_guide_the_search(self):
        image = fixture(ticks=False, grid=True)
        image[20, 31:191] = 1
        image[20:130, 190] = 1
        result = detect_ticks_and_grid(image)
        self.assertIsNotNone(result)
        self.assertEqual(result.frame.rectangle, FRAME.rectangle)
        self.assert_positions(result.vertical_grid, X)
        self.assert_positions(result.horizontal_grid, Y)

    def test_invalid_gap_tolerances_are_rejected(self):
        for gap in (-1, 1.5, True):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                detect_ticks_and_grid(fixture(), frame=FRAME, grid_max_gap=gap)
