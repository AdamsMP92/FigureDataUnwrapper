"""Pixel fixtures for frame-relative tick branches and grid candidates."""

import unittest

import numpy as np

from figure_data_unwrapper import (
    FrameDetection, Rectangle, detect_ticks_and_grid,
)


FRAME = FrameDetection(Rectangle(30, 20, 190, 130), ("left", "top", "right", "bottom"))
X = (50, 90, 150)
Y = (40, 75, 105)


def fixture(*, ticks=True, grid=False, direction="out", opposite=False):
    image = np.ones((160, 220))
    image[20, 30:191] = image[130, 30:191] = 0
    image[20:131, 30] = image[20:131, 190] = 0
    if grid:
        image[21:130, X] = 0.8
        image[np.array(Y)[:, None], np.arange(31, 190)] = 0.8
    if ticks:
        if direction in ("out", "inout"):
            image[131:137, X] = 0
            image[np.array(Y)[:, None], np.arange(24, 30)] = 0
            if opposite:
                image[14:20, X] = 0
                image[np.array(Y)[:, None], np.arange(191, 197)] = 0
        if direction in ("in", "inout"):
            image[124:130, X] = 0
            image[np.array(Y)[:, None], np.arange(31, 37)] = 0
            if opposite:
                image[21:27, X] = 0
                image[np.array(Y)[:, None], np.arange(184, 190)] = 0
    return image


class TickGridTests(unittest.TestCase):
    def assert_positions(self, actual, expected):
        np.testing.assert_allclose([item.position for item in actual], expected, atol=1, rtol=0)

    def test_auto_frame_then_ticks_and_grid(self):
        result = detect_ticks_and_grid(fixture(grid=True))
        self.assertIsNotNone(result)
        self.assertEqual(result.frame.rectangle, FRAME.rectangle)
        self.assert_positions(result.x_ticks, X)
        self.assert_positions(result.y_ticks, Y)
        self.assert_positions(result.vertical_grid, X)
        self.assert_positions(result.horizontal_grid, Y)

    def test_inward_outward_and_both_directions_on_opposing_sides(self):
        for direction in ("in", "out", "inout"):
            for grid in (False, True):
                with self.subTest(direction=direction, grid=grid):
                    result = detect_ticks_and_grid(
                        fixture(direction=direction, grid=grid, opposite=True), frame=FRAME
                    )
                    self.assert_positions(result.x_ticks, X)
                    self.assert_positions(result.y_ticks, Y)
                    for tick in result.x_ticks:
                        self.assertEqual(set(tick.sides), {"bottom", "top"})
                    for tick in result.y_ticks:
                        self.assertEqual(set(tick.sides), {"left", "right"})

    def test_grid_does_not_create_ticks(self):
        result = detect_ticks_and_grid(fixture(ticks=False, grid=True), frame=FRAME)
        self.assertEqual(result.x_ticks, ())
        self.assertEqual(result.y_ticks, ())
        self.assert_positions(result.vertical_grid, X)
        self.assert_positions(result.horizontal_grid, Y)

    def test_frame_without_features_is_empty(self):
        result = detect_ticks_and_grid(fixture(ticks=False), frame=FRAME)
        self.assertEqual((result.x_ticks, result.y_ticks, result.vertical_grid, result.horizontal_grid),
                         ((), (), (), ()))

    def test_dashed_grid_is_separate_from_short_ticks(self):
        image = fixture(ticks=False)
        for x in X:
            for y in range(22, 129, 7):
                image[y:y + 3, x] = 0.8
        for y in Y:
            for x in range(32, 189, 7):
                image[y, x:x + 3] = 0.8
        result = detect_ticks_and_grid(image, frame=FRAME)
        self.assert_positions(result.vertical_grid, X)
        self.assert_positions(result.horizontal_grid, Y)
        self.assertEqual((result.x_ticks, result.y_ticks), ((), ()))

    def test_endpoint_ticks_outside_frame_are_observable(self):
        image = fixture(ticks=False)
        image[131:137, [30, 190]] = 0
        image[[20, 130], 24:30] = 0
        result = detect_ticks_and_grid(image, frame=FRAME)
        self.assert_positions(result.x_ticks, (30, 190))
        self.assert_positions(result.y_ticks, (20, 130))
        self.assertEqual((result.vertical_grid, result.horizontal_grid), ((), ()))

    def test_labels_and_interior_blobs_do_not_become_branches(self):
        image = fixture(ticks=False)
        image[140:147, 48:53] = 0  # Detached label-like stroke.
        image[40:45, 192:196] = 0  # Broad blob touching the right edge.
        image[60:67, 100:107] = 0
        result = detect_ticks_and_grid(image, frame=FRAME)
        self.assertEqual((result.x_ticks, result.y_ticks), ((), ()))

    def test_count_hint_selects_longer_marks(self):
        image = fixture(ticks=False)
        positions = (45, 65, 85, 105, 125, 145, 165)
        image[131:134, positions] = 0
        image[131:139, [45, 105, 165]] = 0
        automatic = detect_ticks_and_grid(image, frame=FRAME)
        self.assert_positions(automatic.x_ticks, positions)
        guided = detect_ticks_and_grid(image, frame=FRAME, expected_x_ticks=3)
        self.assert_positions(guided.x_ticks, (45, 105, 165))
        self.assertTrue(guided.x_count_matches)
        self.assertTrue(guided.notes)
        self.assertIsNone(guided.y_count_matches)

    def test_count_hints_do_not_invent_or_arbitrarily_discard_ticks(self):
        for count in (0, 2, 5):
            with self.subTest(count=count):
                result = detect_ticks_and_grid(fixture(), frame=FRAME, expected_x_ticks=count)
                self.assert_positions(result.x_ticks, X)
                self.assertFalse(result.x_count_matches)
                self.assertTrue(result.notes)

    def test_irregular_positions_are_preserved(self):
        image = fixture(ticks=False)
        positions = (40, 47, 70, 125, 180)
        image[131:137, positions] = 0
        result = detect_ticks_and_grid(image, frame=FRAME)
        self.assert_positions(result.x_ticks, positions)

    def test_frame_failure_gates_extraction(self):
        self.assertIsNone(detect_ticks_and_grid(np.ones((160, 220))))

    def test_invalid_parameters_and_frame_are_rejected(self):
        for kwargs in (
            {"expected_x_ticks": -1}, {"expected_y_ticks": 2.5},
            {"expected_x_ticks": True}, {"min_tick_length": 0},
            {"min_contrast": 1}, {"grid_min_coverage": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detect_ticks_and_grid(fixture(), **kwargs)
        with self.assertRaises(TypeError):
            detect_ticks_and_grid(fixture(), frame=FRAME.rectangle)
        with self.assertRaises(ValueError):
            detect_ticks_and_grid(fixture(), frame=FrameDetection(Rectangle(0, 0, 250, 100),
                                                                 ("left", "bottom")))
