"""Independent marker fixtures with known centers, colors and occlusions."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from matplotlib.image import imread
import numpy as np

from figure_data_unwrapper import (
    AxisTick, FrameDetection, Rectangle, TickGridDetection, check_tick_consistency,
    extract_markers, plot_marker_data, pre_analyzer,
)


def fixture():
    image = np.ones((90, 120, 3))
    frame = FrameDetection(Rectangle(5, 5, 114, 84), ("left", "bottom"))
    raw = TickGridDetection(frame, tuple(AxisTick(p, ()) for p in (15, 55, 95)),
                            tuple(AxisTick(p, ()) for p in (15, 45, 75)), (), (), None, None, ())
    ticks = check_tick_consistency(raw, x_ticks=(0, 1, 2), y_ticks=(0, 1, 2))
    return image, frame, ticks, []


def draw_circle(image, x, y, radius=6, color=(0, 0, 0)):
    yy, xx = np.indices(image.shape[:2])
    image[(xx - x)**2 + (yy - y)**2 <= radius**2] = color


def reference(color=(0, 0, 0)):
    image = np.ones((21, 21, 3))
    draw_circle(image, 10, 10, color=color)
    return image


class MarkerExtractionTests(unittest.TestCase):
    def test_two_isolated_markers_give_one_center_each_and_calibrated_values(self):
        analysis = fixture()
        centers = np.array([[28, 25], [83, 53]])
        for x, y in centers:
            draw_circle(analysis[0], x, y)
        original = analysis[0].copy()
        cloud = extract_markers(analysis, reference())
        self.assertEqual(len(cloud.matches), 2)
        np.testing.assert_allclose(cloud.pixels, centers, atol=.01)
        x, y, eps = cloud
        np.testing.assert_allclose(x, (centers[:, 0] - 15) / 40)
        np.testing.assert_allclose(y, (75 - centers[:, 1]) / 30)
        self.assertTrue((eps > 0).all())
        self.assertTrue(all(m.score > .99 for m in cloud.matches))
        np.testing.assert_array_equal(analysis[0], original)
        self.assertEqual(cloud.to_array().shape, (2, 13))

    def test_reference_padding_does_not_move_the_reported_center(self):
        analysis = fixture()
        draw_circle(analysis[0], 45, 35)
        padded = np.ones((33, 37, 3))
        draw_circle(padded, 13, 21)
        cloud = extract_markers(analysis, padded)
        np.testing.assert_allclose(cloud.pixels, [[45, 35]], atol=.01)
        anchored = extract_markers(analysis, padded, anchor=(14, 21))
        np.testing.assert_allclose(anchored.pixels, [[46, 35]], atol=.01)

    def test_different_color_overdraw_is_masked_and_does_not_shift_center(self):
        analysis = fixture()
        draw_circle(analysis[0], 45, 35)
        analysis[0][34:37, 10:100] = (1, 0, 0)
        cloud = extract_markers(analysis, reference())
        np.testing.assert_allclose(cloud.pixels, [[45, 35]], atol=.5)
        self.assertLess(cloud.matches[0].visible_fraction, .9)
        self.assertGreater(cloud.matches[0].score, .99)

    def test_touching_markers_are_not_collapsed_into_one_connected_blob(self):
        analysis = fixture()
        for x, y in ((40, 40), (50, 40)):
            draw_circle(analysis[0], x, y)
        cloud = extract_markers(analysis, reference())
        self.assertEqual(len(cloud.matches), 2)
        np.testing.assert_allclose(cloud.pixels, [[40, 40], [50, 40]], atol=1)
        self.assertTrue(all(m.overlaps_neighbor for m in cloud.matches))

    def test_scale_search_handles_a_larger_data_marker_than_the_reference(self):
        analysis = fixture()
        draw_circle(analysis[0], 50, 35, radius=7.5)
        cloud = extract_markers(analysis, reference(), scales=(1., 1.25))
        self.assertEqual(len(cloud.matches), 1)
        np.testing.assert_allclose(cloud.pixels, [[50, 35]], atol=1)
        self.assertEqual(cloud.matches[0].scale, 1.25)

    def test_template_color_can_be_inferred_or_explicit(self):
        analysis = fixture()
        draw_circle(analysis[0], 45, 35, color=(0, 0, 1))
        for color in (None, "blue"):
            with self.subTest(color=color):
                cloud = extract_markers(analysis, reference((0, 0, 1)), color=color)
                np.testing.assert_allclose(cloud.pixels, [[45, 35]], atol=.1)
                self.assertEqual(cloud.color, (0, 0, 1))

    def test_thin_error_bars_axes_and_large_filled_regions_are_not_markers(self):
        for kind in ("cross", "block", "square"):
            analysis = fixture()
            if kind == "cross":
                analysis[0][12:78, 45] = analysis[0][40, 12:105] = 0
            elif kind == "block":
                analysis[0][20:60, 20:95] = 0
            else:
                analysis[0][29:42, 39:52] = 0
            with self.subTest(kind=kind):
                cloud = extract_markers(analysis, reference(), min_score=.9)
                self.assertEqual(len(cloud.matches), 0)

    def test_identical_legend_markers_require_explicit_context(self):
        analysis = fixture()
        draw_circle(analysis[0], 35, 25)
        draw_circle(analysis[0], 80, 65)
        all_matches = extract_markers(analysis, reference())
        excluded = Rectangle(65, 50, 95, 78)
        filtered = extract_markers(analysis, reference(), exclude_regions=(excluded,))
        self.assertEqual(len(all_matches.matches), 2)
        np.testing.assert_allclose(filtered.pixels, [[35, 25]], atol=.01)
        self.assertEqual(filtered.excluded_regions, (excluded,))

    def test_ambiguous_ticks_do_not_produce_numeric_marker_coordinates(self):
        analysis = fixture()
        ticks = replace(analysis[2], x_validation=replace(analysis[2].x_validation, status="ambiguous_values"))
        with self.assertRaises(ValueError):
            extract_markers((analysis[0], analysis[1], ticks, []), reference())

    def test_integer_rgba_and_file_templates(self):
        analysis = fixture()
        draw_circle(analysis[0], 45, 35)
        ref = reference()
        for source in ((ref * 255).astype(np.uint8), np.dstack((ref, np.ones(ref.shape[:2]))), ref[..., 0]):
            with self.subTest(shape=source.shape):
                cloud = extract_markers(analysis, source)
                np.testing.assert_allclose(cloud.pixels, [[45, 35]], atol=.01)

    def test_empty_results_have_stable_array_shapes_and_can_be_plotted(self):
        analysis = fixture()
        cloud = extract_markers(analysis, reference())
        self.assertEqual(cloud.pixels.shape, (0, 2))
        self.assertEqual(cloud.eps.shape, (0, 2))
        self.assertEqual(cloud.to_array().shape, (0, 13))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markers.png"
            self.assertEqual(plot_marker_data(analysis, cloud, output=path), path)
            self.assertEqual(imread(path).shape[:2], (825, 1950))

    def test_candidate_limit_is_bounded_and_reported(self):
        analysis = fixture()
        draw_circle(analysis[0], 30, 25)
        draw_circle(analysis[0], 80, 60)
        cloud = extract_markers(analysis, reference(), max_candidates=1)
        self.assertEqual(cloud.candidate_count, 1)
        self.assertEqual(len(cloud.matches), 1)
        self.assertTrue(any("capped" in note for note in cloud.notes))

    def test_noisy_marker_keeps_its_center_and_does_not_get_perfect_score(self):
        analysis = fixture()
        draw_circle(analysis[0], 45, 35)
        noise = np.random.default_rng(17).uniform(-.03, .03, analysis[0].shape[:2])
        analysis[0][:] = np.clip(analysis[0] + noise[..., None], 0, 1)
        cloud = extract_markers(analysis, reference())
        np.testing.assert_allclose(cloud.pixels, [[45, 35]], atol=.5)
        self.assertLess(cloud.matches[0].score, 1)

    def test_bad_configuration_and_blank_reference_fail_explicitly(self):
        for options in ({"scales": []}, {"scales": [0]}, {"min_score": 2}, {"min_distance": 0},
                        {"refinement_steps": 6}, {"max_candidates": True}, {"color_tolerance": 0},
                        {"anchor": (100, 0)}, {"frame_margin": 50}, {"exclude_regions": [(1, 2, 3, 4)]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                extract_markers(fixture(), reference(), **options)
        with self.assertRaises(ValueError):
            extract_markers(fixture(), np.ones((21, 21, 3)))

    def test_real_paper_reference_outputs_checkable_marker_hypotheses(self):
        directory = Path(__file__).resolve().parents[1] / "examples" / "real_example_1"
        with tempfile.TemporaryDirectory() as temporary:
            analysis = pre_analyzer(directory / "real_example1.png", 3, output=Path(temporary) / "precheck.png",
                                    x_ticks=np.linspace(0, 360, 17), y_ticks=[0, 2.5, 5, 7.5, 10])
        excluded = Rectangle(200, 480, 320, 580)
        cloud = extract_markers(analysis, directory / "black_marker.png", scales=(1., 1.1, 1.2),
                                exclude_regions=(excluded,))
        # No unavailable paper dataset is treated as ground truth. Verify only
        # visible broad coverage, valid numeric mapping and explicit exclusions.
        self.assertGreater(len(cloud.matches), 25)
        self.assertGreater(cloud.x.max() - cloud.x.min(), 340)
        self.assertGreater(cloud.y.min(), 3.5)
        self.assertLess(cloud.y.max(), 10)
        self.assertFalse(any(excluded.left <= m.pixel_x <= excluded.right and
                             excluded.top <= m.pixel_y <= excluded.bottom for m in cloud.matches))
        self.assertTrue(all(m.score >= .65 for m in cloud.matches))
        np.testing.assert_allclose(analysis[2].x_validation.pixel_slope * cloud.x
                                   + analysis[2].x_validation.pixel_offset, cloud.pixels[:, 0])
        np.testing.assert_allclose(analysis[2].y_validation.pixel_slope * cloud.y
                                   + analysis[2].y_validation.pixel_offset, cloud.pixels[:, 1])
