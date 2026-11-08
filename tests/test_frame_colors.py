"""Original-RGB frame-path validation and transition-supported correction."""

from pathlib import Path
import unittest

from matplotlib.image import imread
import numpy as np

from figure_data_unwrapper import FrameDetection, Rectangle, check_frame_colors, detect_ticks_and_grid


SIDES = ("left", "top", "right", "bottom")
BOUNDS = Rectangle(20, 20, 210, 150)


def image_with_sides(colors=None, background=0.5):
    colors = colors or {side: (0, 0, 0) for side in SIDES}
    image = np.full((180, 240, 3), background)
    for side, color in colors.items():
        if side == "left":
            image[20:151, 20] = color
        elif side == "right":
            image[20:151, 210] = color
        elif side == "top":
            image[20, 20:211] = color
        else:
            image[150, 20:211] = color
    return image


def shadowed_image():
    image = np.ones((180, 240, 3))
    image[:25] = (0.85 + 0.005 * np.arange(25))[:, None, None]
    image[20, 20:211] = image[150, 20:211] = 0
    image[20:151, 20] = image[20:151, 210] = 0
    return image


class FrameColorTests(unittest.TestCase):
    def test_each_side_can_have_its_own_color_including_white(self):
        colors = {"left": (0, 0, 0), "top": (1, 0, 0),
                  "right": (1, 1, 1), "bottom": (0, 0, 1)}
        image = image_with_sides(colors)
        original = image.copy()
        checked = check_frame_colors(image, FrameDetection(BOUNDS, SIDES))
        self.assertEqual(checked.rectangle, BOUNDS)
        self.assertFalse(checked.color_check.corrected)
        self.assertEqual(checked.color_check.status, "valid")
        for edge in checked.color_check.edges:
            with self.subTest(side=edge.side):
                np.testing.assert_allclose(edge.color, colors[edge.side])
                self.assertTrue(edge.plausible)
        np.testing.assert_array_equal(image, original)

    def test_shadowed_top_is_corrected_using_adjacent_color_transitions(self):
        proposed = FrameDetection(Rectangle(20, 0, 210, 150), ("left", "right", "bottom"))
        checked = check_frame_colors(shadowed_image(), proposed, search_fraction=0.2)
        self.assertEqual(checked.rectangle, BOUNDS)
        self.assertTrue(checked.color_check.corrected)
        self.assertEqual(checked.color_check.original_rectangle, proposed.rectangle)
        self.assertEqual(checked.color_check.status, "valid")
        self.assertEqual(set(checked.visible_sides), set(SIDES))
        for side in ("left", "right"):
            before = next(edge for edge in checked.color_check.original_edges if edge.side == side)
            self.assertTrue(any(not change.allowed and change.reason == "endpoint_color_change"
                                for change in before.changes))
            self.assertIn(20, before.jump_positions)

    def test_diagnostics_can_be_requested_without_correcting(self):
        proposed = FrameDetection(Rectangle(20, 0, 210, 150), ("left", "right", "bottom"))
        checked = check_frame_colors(shadowed_image(), proposed, correct=False)
        self.assertEqual(checked.rectangle, proposed.rectangle)
        self.assertFalse(checked.color_check.corrected)
        self.assertEqual(checked.color_check.status, "suspect")

    def test_transition_correction_is_symmetric_for_all_four_boundaries(self):
        image = shadowed_image()
        proposed, expected = Rectangle(20, 0, 210, 150), BOUNDS
        for quarter_turns in range(4):
            with self.subTest(quarter_turns=quarter_turns):
                checked = check_frame_colors(image, FrameDetection(proposed, SIDES), search_fraction=0.2)
                self.assertEqual(checked.rectangle, expected)
                self.assertEqual(checked.color_check.status, "valid")
            width = image.shape[1]
            proposed = Rectangle(proposed.top, width - 1 - proposed.right,
                                 proposed.bottom, width - 1 - proposed.left)
            expected = Rectangle(expected.top, width - 1 - expected.right,
                                 expected.bottom, width - 1 - expected.left)
            image = np.rot90(image)

    def test_short_crossing_and_bounded_occlusion_are_allowed(self):
        for interval, reason in ((slice(60, 62), "short_interruption"),
                                 (slice(70, 83), "bounded_occlusion")):
            with self.subTest(reason=reason):
                image = image_with_sides()
                image[interval, 20] = (0, 1, 0)
                checked = check_frame_colors(image, FrameDetection(BOUNDS, SIDES))
                left = next(edge for edge in checked.color_check.edges if edge.side == "left")
                self.assertEqual(left.status, "interrupted")
                self.assertTrue(all(change.allowed for change in left.changes))
                self.assertIn(reason, [change.reason for change in left.changes])
                self.assertFalse(checked.color_check.corrected)

    def test_persistent_change_within_one_side_is_flagged(self):
        image = image_with_sides()
        image[85:151, 20] = 1
        checked = check_frame_colors(image, FrameDetection(BOUNDS, SIDES))
        left = next(edge for edge in checked.color_check.edges if edge.side == "left")
        self.assertEqual(left.status, "color_change")
        self.assertTrue(any(not change.allowed for change in left.changes))
        self.assertEqual(checked.color_check.status, "suspect")
        self.assertFalse(checked.color_check.corrected)

    def test_long_endpoint_loss_is_not_silently_treated_as_a_missing_side(self):
        image = image_with_sides(background=1.)
        image[100:149, 20] = 1
        checked = check_frame_colors(image, FrameDetection(BOUNDS, SIDES))
        left = next(edge for edge in checked.color_check.edges if edge.side == "left")
        self.assertGreater(left.stroke_fraction, 0.5)
        self.assertLess(left.stroke_fraction, 0.75)
        self.assertEqual(left.status, "color_change")
        self.assertEqual(checked.color_check.status, "suspect")
        self.assertFalse(checked.color_check.corrected)

    def test_constant_background_is_not_line_evidence(self):
        checked = check_frame_colors(np.ones((180, 240, 3)), FrameDetection(BOUNDS, SIDES))
        self.assertTrue(all(edge.consistent_fraction == 1 for edge in checked.color_check.edges))
        self.assertTrue(all(edge.stroke_fraction == 0 for edge in checked.color_check.edges))
        self.assertEqual(checked.visible_sides, ())
        self.assertEqual(checked.color_check.status, "rejected")

    def test_smooth_background_gradient_is_not_a_stroke(self):
        y, x = np.indices((180, 240))
        image = np.repeat((0.2 + 0.2 * x / 240 + 0.2 * y / 180)[..., None], 3, axis=2)
        checked = check_frame_colors(image, FrameDetection(BOUNDS, SIDES))
        self.assertTrue(all(edge.stroke_fraction == 0 for edge in checked.color_check.edges))
        self.assertEqual(checked.color_check.status, "rejected")

    def test_unresolved_frames_do_not_seed_ticks_or_grid(self):
        suspect_image = image_with_sides()
        suspect_image[85:151, 20] = 1
        for image in (suspect_image, np.ones((180, 240, 3))):
            checked = check_frame_colors(image, FrameDetection(BOUNDS, SIDES))
            with self.subTest(status=checked.color_check.status):
                self.assertIsNone(detect_ticks_and_grid(image, frame=checked))

    def test_missing_sides_are_not_invented_from_uniform_background(self):
        image = image_with_sides({"left": (0, 0, 0), "bottom": (0, 0, 1)})
        checked = check_frame_colors(image, FrameDetection(BOUNDS, ("left", "bottom")))
        self.assertEqual(checked.rectangle, BOUNDS)
        self.assertEqual(set(checked.visible_sides), {"left", "bottom"})
        self.assertEqual(checked.color_check.status, "partial")
        self.assertFalse(checked.color_check.corrected)

    def test_small_rgb_noise_does_not_move_valid_edges(self):
        image = image_with_sides({side: (0.2, 0.2, 0.2) for side in SIDES}, background=1.)
        rng = np.random.default_rng(18)
        image = np.clip(image + rng.uniform(-0.01, 0.01, image.shape), 0, 1)
        checked = check_frame_colors(image, FrameDetection(BOUNDS, SIDES))
        self.assertEqual(checked.rectangle, BOUNDS)
        self.assertEqual(checked.color_check.status, "valid")

    def test_integer_grayscale_rgba_and_boolean_inputs(self):
        image = image_with_sides(background=1.)
        sources = ((image * 255).astype(np.uint8), image[..., 0],
                   np.dstack((image, np.ones(image.shape[:2]))), image[..., 0] == 0)
        for source in sources:
            with self.subTest(shape=source.shape, dtype=source.dtype):
                checked = check_frame_colors(source, FrameDetection(BOUNDS, SIDES))
                self.assertEqual(checked.rectangle, BOUNDS)
                self.assertEqual(checked.color_check.status, "valid")

    def test_search_range_limits_unsupported_large_moves(self):
        proposed = FrameDetection(Rectangle(20, 0, 210, 150), ("left", "right", "bottom"))
        for fraction in (0.01, 0.1):
            with self.subTest(search_fraction=fraction):
                checked = check_frame_colors(shadowed_image(), proposed, search_fraction=fraction)
                self.assertEqual(checked.rectangle, proposed.rectangle)
                self.assertEqual(checked.color_check.status, "suspect")

    def test_real_example2_corrects_top_without_sharing_side_colors(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "real_example_2" / "real_example2.png"
        proposed = FrameDetection(Rectangle(228, 0, 1661, 1459), ("left", "right", "bottom"))
        checked = check_frame_colors(imread(path), proposed)
        self.assertAlmostEqual(checked.rectangle.top, 27, delta=1)
        self.assertEqual((checked.rectangle.left, checked.rectangle.right, checked.rectangle.bottom),
                         (228, 1661, 1459))
        self.assertTrue(checked.color_check.corrected)
        self.assertEqual(checked.color_check.status, "valid")

    def test_invalid_configuration(self):
        for options in ({"color_tolerance": 0}, {"probe_radius": 0},
                        {"max_interruption_px": -1}, {"search_fraction": 0.6}, {"correct": "yes"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                check_frame_colors(np.ones((10, 10, 3)), None, **options)
