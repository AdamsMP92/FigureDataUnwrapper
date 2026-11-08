"""Independent partial-frame checks without grid lines or renderer metadata."""

from itertools import combinations
import unittest

import numpy as np

from figure_data_unwrapper import Rectangle, detect_frame, fit_frame, foreground_mask


def draw_sides(image, bounds, sides, value, thickness=1):
    """Use direct slices as fixtures, independently of the rectangle model."""
    left, top, right, bottom = bounds
    radius = thickness // 2
    if "left" in sides:
        image[top:bottom + 1, left - radius:left + radius + 1] = value
    if "right" in sides:
        image[top:bottom + 1, right - radius:right + radius + 1] = value
    if "top" in sides:
        image[top - radius:top + radius + 1, left:right + 1] = value
    if "bottom" in sides:
        image[bottom - radius:bottom + radius + 1, left:right + 1] = value


class PartialFrameTests(unittest.TestCase):
    def test_all_reconstructable_side_combinations_without_grid(self):
        rng = np.random.default_rng(42)
        for count in (2, 3, 4):
            for sides in combinations(("left", "top", "right", "bottom"), count):
                for thickness, value in ((1, 0.05), (3, 0.8), (5, 0.93)):
                    with self.subTest(sides=sides, thickness=thickness, value=value):
                        left, top = (int(value) for value in rng.integers(12, 28, size=2))
                        right, bottom = (int(value) for value in rng.integers(110, 130, size=2))
                        bounds = (left, top, right, bottom)
                        image = np.ones((150, 170))
                        draw_sides(image, bounds, sides, value, thickness)
                        self.assertEqual(detect_frame(image), Rectangle(*bounds))

    def test_measured_and_inferred_sides_are_distinguished(self):
        image = np.ones((120, 160))
        draw_sides(image, (20, 15, 140, 105), ("left", "bottom"), 0.9)
        result = fit_frame(foreground_mask(image))
        self.assertIsNotNone(result)
        self.assertEqual(result.rectangle, Rectangle(20, 15, 140, 105))
        self.assertEqual(set(result.visible_sides), {"left", "bottom"})
        self.assertEqual(set(result.inferred_sides), {"top", "right"})

    def test_threshold_perturbations_preserve_geometry(self):
        image = np.ones((120, 160))
        draw_sides(image, (20, 15, 140, 105), ("left", "bottom"), 0.8)
        for threshold in (0.02, 0.04, 0.08):
            with self.subTest(threshold=threshold):
                result = fit_frame(foreground_mask(image, min_contrast=threshold))
                self.assertIsNotNone(result)
                self.assertEqual(result.rectangle, Rectangle(20, 15, 140, 105))

    def test_white_partial_frame_on_dark_background(self):
        image = np.zeros((120, 160))
        draw_sides(image, (20, 15, 140, 105), ("top", "right"), 0.9)
        self.assertEqual(detect_frame(image), Rectangle(20, 15, 140, 105))

    def test_sides_with_different_colors(self):
        image = np.ones((120, 160, 3))
        for side, color in zip(
            ("left", "top", "right", "bottom"),
            ([1, 1, 0], [0, 1, 1], [0.8, 0.8, 0.8], [1, 0.5, 0]),
        ):
            draw_sides(image, (20, 15, 140, 105), (side,), color)
        self.assertEqual(detect_frame(image), Rectangle(20, 15, 140, 105))

    def test_crossing_lines_and_misaligned_parallel_lines_do_not_define_frame(self):
        image = np.ones((120, 160))
        image[15:106, 80] = 0
        image[60, 20:141] = 0
        self.assertIsNone(detect_frame(image))
        image = np.ones((120, 160))
        image[20, 10:101] = 0
        image[100, 40:151] = 0
        self.assertIsNone(detect_frame(image))

    def test_filled_region_is_not_a_thick_frame(self):
        image = np.ones((120, 160))
        image[15:106, 20:141] = 0
        self.assertIsNone(detect_frame(image))

    def test_fixed_threshold_experiment_is_available(self):
        image = np.ones((120, 160))
        draw_sides(image, (20, 15, 140, 105), ("left", "bottom"), 0.93)
        self.assertIsNone(detect_frame(image, darkness_threshold=0.5))
        self.assertEqual(detect_frame(image), Rectangle(20, 15, 140, 105))

    def test_fit_requires_a_boolean_mask(self):
        with self.assertRaises(ValueError):
            fit_frame(np.ones((20, 30)))
