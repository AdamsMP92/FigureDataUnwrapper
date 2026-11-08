"""Color selection, calibrated coordinates and model-based uncertainty."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from matplotlib.image import imread
import numpy as np

from figure_data_unwrapper import (
    AxisTick, FrameColorValidation, FrameDetection, Rectangle, TickGridDetection,
    calibrate_axes, check_tick_consistency, extract_color_data, plot_color_data, pre_analyzer,
)


def analysis(x=(20, 40, 60), x_values=(-1, 0, 1), y=(10, 25, 40), y_values=(0, 1, 2), **options):
    image = np.ones((60, 80, 3))
    frame = FrameDetection(Rectangle(10, 5, 70, 45), ("left", "bottom"))
    raw = TickGridDetection(frame, tuple(AxisTick(p, ()) for p in x),
                            tuple(AxisTick(p, ()) for p in y), (), (), None, None, ())
    ticks = check_tick_consistency(raw, x_ticks=x_values, y_ticks=y_values, **options)
    return image, frame, ticks, []


class ColorExtractionTests(unittest.TestCase):
    def test_pixel_centers_convert_to_axis_values_without_changing_source(self):
        result = analysis()
        result[0][25, 40] = result[0][10, 20] = (1, 0, 0)
        original = result[0].copy()
        cloud = extract_color_data(result, "red")
        x, y, eps = cloud
        np.testing.assert_allclose(x, [-1, 0])
        np.testing.assert_allclose(y, [2, 1])
        np.testing.assert_array_equal(cloud.pixels, [[20, 10], [40, 25]])
        self.assertEqual(eps.shape, (2, 2))
        self.assertTrue((eps > 0).all())
        np.testing.assert_array_equal(result[0], original)
        np.testing.assert_array_equal(cloud.to_array()[:, :2], np.column_stack((x, y)))
        self.assertEqual(cloud.to_array().shape, (2, 6))

    def test_exact_calibration_has_resolution_floor_and_known_covariance(self):
        calibrated = calibrate_axes(analysis()[2])
        x, eps = calibrated.x.transform([40, 60, 80])
        np.testing.assert_allclose(x, [0, 1, 2])
        sigma = 1 / np.sqrt(12)
        np.testing.assert_allclose(eps, sigma / 20 * np.sqrt([1/3, 1/3 + 1/2, 1/3 + 4/2]))
        self.assertGreater(eps[2], eps[1])
        _, doubled = calibrate_axes(analysis()[2], tick_sigma_px=2 * sigma).x.transform([40])
        np.testing.assert_allclose(doubled, 2 * eps[:1])

    def test_residual_variance_increases_uncertainty(self):
        exact = calibrate_axes(analysis(x=(15, 25, 35, 45, 55), x_values=(0, 1, 2, 3, 4))[2])
        noisy = calibrate_axes(analysis(x=(15, 25, 36, 45, 55), x_values=(0, 1, 2, 3, 4))[2])
        self.assertGreater(noisy.x.residual_sigma_pixels, 1 / np.sqrt(12))
        self.assertGreater(noisy.x.transform([35])[1][0], exact.x.transform([35])[1][0])

    def test_logarithmic_inverse_and_uncertainty_use_axis_units(self):
        calibration = calibrate_axes(analysis(x=(10, 30, 50), x_values=(1, 10, 100), x_log=True)[2])
        x, eps = calibration.x.transform([10, 30, 50])
        np.testing.assert_allclose(x, [1, 10, 100])
        expected_u_eps = (1 / np.sqrt(12)) / (20 / np.log(10)) * np.sqrt([5/6, 1/3, 5/6])
        np.testing.assert_allclose(eps, x * expected_u_eps)
        np.testing.assert_allclose(eps[0] / x[0], eps[2] / x[2])

    def test_reversed_axes_are_preserved(self):
        calibration = calibrate_axes(analysis(x_values=(1, 0, -1), y_values=(2, 1, 0))[2])
        np.testing.assert_allclose(calibration.x.transform([20, 60])[0], [1, -1])
        np.testing.assert_allclose(calibration.y.transform([10, 40])[0], [0, 2])
        self.assertTrue((calibration.x.transform([20, 60])[1] > 0).all())

    def test_loglog_cloud_calibrates_both_coordinates(self):
        result = analysis(x_values=(1, 10, 100), y_values=(1, 10, 100), x_log=True, y_log=True)
        result[0][25, 40] = (1, 0, 0)
        cloud = extract_color_data(result, "red")
        np.testing.assert_allclose(cloud.x, [10])
        np.testing.assert_allclose(cloud.y, [10])
        self.assertTrue((cloud.eps > 0).all())

    def test_two_anchor_mapping_does_not_claim_zero_uncertainty(self):
        calibration = calibrate_axes(analysis(x=(20, 60), x_values=(-1, 1))[2])
        self.assertEqual(calibration.x.anchor_count, 2)
        self.assertGreater(calibration.x.transform([40])[1][0], 0)
        self.assertTrue(any("two anchors" in note for note in calibration.notes))

    def test_unlabelled_or_ambiguous_axes_cannot_be_calibrated(self):
        for result in (analysis(x_values=None),
                       analysis(x=(20, 30, 40, 50), x_values=(0, 1, 2, 3, 4))):
            with self.subTest(status=result[2].x_validation), self.assertRaises(ValueError):
                extract_color_data(result, "red")

    def test_largest_component_removes_separate_legend_but_all_keeps_markers(self):
        result = analysis()
        result[0][20, 20:61] = (1, 0, 0)
        result[0][35, 20:31] = (1, 0, 0)
        all_pixels = extract_color_data(result, "red")
        curve = extract_color_data(result, "red", components="largest")
        self.assertEqual(len(all_pixels.x), 52)
        self.assertEqual(len(curve.x), 41)
        self.assertEqual(curve.component_count, 2)
        self.assertEqual(curve.matched_pixel_count, 52)
        self.assertTrue((curve.pixels[:, 1] == 20).all())
        self.assertTrue(any("excluded 11" in note for note in curve.notes))

    def test_diagonal_neighbors_are_connected_and_branches_are_merged(self):
        result = analysis()
        pixels = [(20, 20), (22, 20), (21, 21), (22, 22), (35, 30)]
        for x, y in pixels:
            result[0][y, x] = (1, 0, 0)
        cloud = extract_color_data(result, "red", components="largest")
        self.assertEqual(cloud.component_count, 2)
        self.assertEqual(len(cloud.x), 4)
        self.assertNotIn((35, 30), [tuple(pixel) for pixel in cloud.pixels])

    def test_white_margin_frame_and_other_colors_are_excluded(self):
        result = analysis()
        result[0][25, 40] = (1, 0, 0)
        result[0][25, 10] = result[0][25, 14] = result[0][50, 40] = (1, 0, 0)
        result[0][25, 41] = (0, 0, 1)
        cloud = extract_color_data(result, "red")
        np.testing.assert_array_equal(cloud.pixels, [[40, 25]])

    def test_rgb_distance_tolerates_nearby_antialiasing_shades(self):
        result = analysis()
        result[0][25, 40:43] = [(1, 0, 0), (1, .1, .1), (1, .4, .4)]
        self.assertEqual(len(extract_color_data(result, "red", color_tolerance=.2).x), 2)
        self.assertEqual(len(extract_color_data(result, "red", color_tolerance=0).x), 1)

    def test_integer_rgba_and_hex_color_conventions(self):
        result = analysis()
        result[0][25, 40] = (1, 0, 0)
        for image, color in ((result[0], "#ff0000"), ((result[0] * 255).astype(np.uint8), (255, 0, 0)),
                             (np.dstack((result[0], np.ones((60, 80)))), (1., 0., 0.))):
            with self.subTest(color=color):
                cloud = extract_color_data((image, *result[1:]), color)
                np.testing.assert_array_equal(cloud.pixels, [[40, 25]])
        transparent = np.zeros((60, 80, 4))
        transparent[..., 0] = 1
        self.assertEqual(len(extract_color_data((transparent, *result[1:]), "red").x), 0)

    def test_no_match_returns_well_shaped_empty_cloud(self):
        cloud = extract_color_data(analysis(), "red", components="largest")
        self.assertEqual(cloud.x.shape, (0,))
        self.assertEqual(cloud.eps.shape, (0, 2))
        self.assertEqual(cloud.pixels.shape, (0, 2))
        self.assertEqual(cloud.to_array().shape, (0, 6))
        self.assertEqual(cloud.component_count, 0)

    def test_bad_options_or_unsuccessful_analysis_fail_explicitly(self):
        result = analysis()
        for options in ({"color_tolerance": -1}, {"color_tolerance": 2}, {"frame_margin": 30},
                        {"frame_margin": np.nan}, {"components": "guess"}, {"tick_sigma_px": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                extract_color_data(result, "red", **options)
        for color in ([256, 0, 0], [1., np.nan, 0.], [1, 0], "not_a_color"):
            with self.subTest(color=color), self.assertRaises(ValueError):
                extract_color_data(result, color)
        for invalid in (None, (result[0], None, None, [])):
            with self.assertRaises(ValueError):
                extract_color_data(invalid, "red")
        suspect = replace(result[1], color_check=FrameColorValidation(result[1].rectangle, (), (), False, "suspect"))
        with self.assertRaises(ValueError):
            extract_color_data((result[0], suspect, result[2], []), "red")
        with self.assertRaises(ValueError):
            calibrate_axes(result[2]).x.transform([np.inf])

    def test_export_plot_and_csv_keep_calibrated_coordinates(self):
        result = analysis()
        result[0][25, 30:50] = (1, 0, 0)
        cloud = extract_color_data(result, "red")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.png"
            self.assertEqual(plot_color_data(result, cloud, output=path), path)
            self.assertEqual(imread(path).shape[:2], (795, 1800))
            csv = path.with_suffix(".csv")
            np.savetxt(csv, cloud.to_array(), delimiter=",")
            np.testing.assert_allclose(np.loadtxt(csv, delimiter=","), cloud.to_array())
            with self.assertRaises(ValueError):
                plot_color_data(result, cloud, output=path.with_suffix(".jpg"))

    def test_real_paper_red_curve_excludes_legend_and_reuses_calibration(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "real_example_1" / "real_example1.png"
        with tempfile.TemporaryDirectory() as directory:
            result = pre_analyzer(path, 3, output=Path(directory) / "precheck.png",
                                  x_ticks=np.linspace(0, 360, 17), y_ticks=[0, 2.5, 5, 7.5, 10])
        cloud = extract_color_data(result, "red", components="largest")
        self.assertIs(cloud.frame, result[1])
        self.assertEqual(cloud.calibration.x.anchor_count, 17)
        self.assertEqual(cloud.calibration.y.anchor_count, 5)
        self.assertEqual(cloud.component_count, 2)
        self.assertGreater(len(cloud.x), 9000)
        self.assertLess(len(cloud.x), cloud.matched_pixel_count)
        self.assertGreater(cloud.x.max() - cloud.x.min(), 350)
        self.assertGreater(cloud.y.min(), 4.5)  # The red legend is near y=1.
        self.assertLess(cloud.y.max(), 9.5)
        self.assertTrue((cloud.eps > 0).all())
        # Every output point remains invertible to its original source pixel.
        np.testing.assert_allclose(result[2].x_validation.pixel_slope * cloud.x
                                   + result[2].x_validation.pixel_offset, cloud.pixels[:, 0])
        np.testing.assert_allclose(result[2].y_validation.pixel_slope * cloud.y
                                   + result[2].y_validation.pixel_offset, cloud.pixels[:, 1])
