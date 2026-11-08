"""Axis-value matching and conservative separation of inconsistent ticks."""

import unittest

import numpy as np

from figure_data_unwrapper import (
    AxisTick, FrameDetection, Rectangle, TickGridDetection, TickMark,
    check_tick_consistency,
)


def detection(x=(), y=()):
    frame = FrameDetection(Rectangle(0, 0, 1000, 1000), ("left", "bottom"))
    return TickGridDetection(
        frame,
        tuple(AxisTick(float(p), (TickMark("bottom", float(p), 0, 6),)) for p in sorted(x)),
        tuple(AxisTick(float(p), (TickMark("left", float(p), 0, 6),)) for p in sorted(y)),
        (), (), None, None, (),
    )


class TickValidationTests(unittest.TestCase):
    def test_manual_linear_values_separate_outliers(self):
        found = detection(x=[20, 70, 99, 120, 170, 198, 220])
        result = check_tick_consistency(found, x_ticks=[-2, -1, 0, 1, 2], x_lin=True)
        self.assertEqual([t.position for t in result.x_ticks], [20, 70, 120, 170, 220])
        self.assertEqual([t.value for t in result.x_ticks], [-2, -1, 0, 1, 2])
        self.assertEqual([r.tick.position for r in result.rejected_x_ticks], [99, 198])
        self.assertTrue(all(r.reason == "off_reference" for r in result.rejected_x_ticks))
        self.assertAlmostEqual(result.x_validation.pixel_slope, 50)
        self.assertAlmostEqual(result.x_validation.pixel_offset, 120)
        self.assertTrue(result.x_count_matches)
        self.assertEqual(len(found.x_ticks), 7)

    def test_y_values_are_bottom_to_top_and_not_frame_limits(self):
        found = detection(y=[31.5, 71, 76.5, 203.5, 377.5, 549.5, 721.5])
        result = check_tick_consistency(found, y_ticks=[0, 2.5, 5, 7.5, 10], y_lin=True)
        self.assertEqual([t.position for t in result.y_ticks], [31.5, 203.5, 377.5, 549.5, 721.5])
        self.assertEqual([t.value for t in result.y_ticks], [10, 7.5, 5, 2.5, 0])
        self.assertEqual([r.tick.position for r in result.rejected_y_ticks], [71, 76.5])
        self.assertLess(result.y_validation.pixel_slope, 0)

    def test_log_values_support_nonuniform_pixel_spacing(self):
        values = np.array([1, 2, 5, 10, 20, 50, 100])
        pixels = 40 + 60 * np.log(values)
        result = check_tick_consistency(detection(x=[*pixels, 70]), x_ticks=values,
                                        x_log=True, tolerance_px=0.25)
        np.testing.assert_allclose([t.position for t in result.x_ticks], pixels)
        np.testing.assert_allclose([t.value for t in result.x_ticks], values)
        self.assertEqual([r.tick.position for r in result.rejected_x_ticks], [70])
        self.assertAlmostEqual(result.x_validation.pixel_slope, 60)

    def test_irregular_linear_values_and_reversed_axes(self):
        values = np.array([0, 1, 3, 8])
        result = check_tick_consistency(detection(x=[*(20 + 15 * values), 60]), x_ticks=values)
        np.testing.assert_allclose([t.value for t in result.x_ticks], values)
        reverse = check_tick_consistency(detection(x=[10, 30, 50, 70]), x_ticks=[10, 5, 0, -5])
        self.assertEqual([t.value for t in reverse.x_ticks], [10, 5, 0, -5])
        self.assertLess(reverse.x_validation.pixel_slope, 0)

    def test_loglog_checks_both_axes_independently(self):
        values = np.array([1, 10, 100, 1000])
        x = 20 + 40 * np.log(values)
        y = 400 - 50 * np.log(values)
        result = check_tick_consistency(detection(x=[*x, 60], y=[*y, 280]),
                                        x_ticks=values, y_ticks=values, x_log=True, y_log=True)
        np.testing.assert_allclose([tick.value for tick in result.x_ticks], values)
        np.testing.assert_allclose([tick.value for tick in result.y_ticks], values[::-1])
        self.assertEqual(len(result.rejected_x_ticks), 1)
        self.assertEqual(len(result.rejected_y_ticks), 1)

    def test_missing_manual_value_is_reported(self):
        result = check_tick_consistency(detection(x=[30, 40, 90, 130]), x_ticks=[0, 1, 3, 6, 10])
        self.assertEqual(result.x_validation.missing_values, (3,))
        self.assertFalse(result.x_count_matches)
        self.assertEqual(len(result.x_ticks), 4)

    def test_ambiguous_value_assignment_does_not_invent_labels(self):
        result = check_tick_consistency(detection(x=[20, 30, 40, 50]), x_ticks=[0, 1, 2, 3, 4])
        self.assertEqual(result.x_validation.status, "ambiguous_values")
        self.assertEqual(len(result.x_ticks), 4)
        self.assertTrue(all(t.value is None for t in result.x_ticks))
        self.assertIsNone(result.x_validation.pixel_slope)

    def test_competing_inlier_sets_leave_observations_untouched(self):
        result = check_tick_consistency(detection(x=[0, 10, 20, 30, 40]), x_ticks=[0, 1, 2])
        self.assertEqual(result.x_validation.status, "ambiguous")
        self.assertEqual(len(result.x_ticks), 5)
        self.assertEqual(result.rejected_x_ticks, ())

    def test_two_values_cannot_isolate_outliers(self):
        result = check_tick_consistency(detection(x=[10, 20, 30]), x_ticks=[0, 1])
        self.assertEqual(result.x_validation.status, "insufficient")
        self.assertEqual(len(result.x_ticks), 3)
        anchored = check_tick_consistency(detection(x=[10, 20]), x_ticks=[0, 1])
        self.assertEqual(anchored.x_validation.status, "two_point")
        self.assertEqual([t.value for t in anchored.x_ticks], [0, 1])

    def test_default_preserves_detection(self):
        found = detection(x=[1, 19, 25, 61])
        self.assertIs(check_tick_consistency(found), found)

    def test_separated_evidence_can_be_rechecked_with_new_values(self):
        positions = [20, 70, 99, 120, 170, 198, 220]
        first = check_tick_consistency(detection(x=positions), x_ticks=[-2, -1, 0, 1, 2])
        second = check_tick_consistency(first, x_ticks=positions)
        self.assertEqual([tick.position for tick in second.x_ticks], positions)
        self.assertEqual(second.rejected_x_ticks, ())
        self.assertEqual([tick.value for tick in second.x_ticks], positions)

    def test_linear_spacing_only_can_separate_clear_outliers(self):
        result = check_tick_consistency(detection(x=[10, 30, 43, 50, 70, 90, 110, 117]), x_lin=True)
        self.assertEqual([t.position for t in result.x_ticks], [10, 30, 50, 70, 90, 110])
        self.assertEqual([r.tick.position for r in result.rejected_x_ticks], [43, 117])
        self.assertAlmostEqual(result.x_validation.spacing_pixels, 20)
        self.assertTrue(all(t.value is None for t in result.x_ticks))

    def test_log_flag_alone_does_not_reject_minor_tick_patterns(self):
        pixels = 20 + 80 * np.log([1, 2, 3, 5, 10, 20, 30, 50])
        result = check_tick_consistency(detection(x=pixels), x_log=True)
        self.assertEqual(result.x_validation.status, "needs_values")
        self.assertEqual(len(result.x_ticks), len(pixels))
        self.assertEqual(result.rejected_x_ticks, ())

    def test_empty_reference_separates_all_observations(self):
        result = check_tick_consistency(detection(x=[10, 20, 30]), x_ticks=[])
        self.assertEqual(result.x_ticks, ())
        self.assertEqual(len(result.rejected_x_ticks), 3)
        self.assertTrue(result.x_count_matches)

    def test_invalid_options_are_rejected_even_without_a_frame(self):
        for options in (
            {"x_lin": True, "x_log": True}, {"y_lin": 1},
            {"x_ticks": [[1, 2]]}, {"x_ticks": [1, 1, 2]},
            {"x_ticks": [1, 3, 2]}, {"y_ticks": [0, np.inf]},
            {"x_ticks": [0, 1], "x_log": True}, {"y_ticks": [-1, 1], "y_log": True},
            {"x_ticks": [0, 1, 2], "expected_x_ticks": 4}, {"tolerance_px": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                check_tick_consistency(None, **options)
