"""Two-band association retains source pixels and rejects unsupported guesses."""

from pathlib import Path
import tempfile
import unittest

from matplotlib.image import imread
import numpy as np

from figure_data_unwrapper import (
    AxisTick, FrameDetection, Rectangle, TickGridDetection, check_tick_consistency,
    extract_color_data, plot_color_curves, pre_analyzer, remove_colors, split_two_curves,
)


def fixture():
    image = np.ones((90, 120, 3))
    frame = FrameDetection(Rectangle(5, 5, 114, 84), ("left", "bottom"))
    raw = TickGridDetection(frame, tuple(AxisTick(p, ()) for p in (15, 55, 95)),
                            tuple(AxisTick(p, ()) for p in (15, 45, 75)), (), (), None, None, ())
    ticks = check_tick_consistency(raw, x_ticks=(0, 1, 2), y_ticks=(0, 1, 2))
    for x in range(20, 91):
        image[15 + (x - 20) // 5, x] = 0
        image[45 + (x - 20) // 5, x] = 0
    return image, frame, ticks, []


class CurveSplitTests(unittest.TestCase):
    def test_two_sloped_curves_partition_the_original_pixels_and_uncertainties(self):
        analysis = fixture()
        all_pixels = extract_color_data(analysis, "black")
        upper, lower = split_two_curves(all_pixels, min_separation_px=20)
        self.assertEqual(len(upper.x), 71)
        self.assertEqual(len(lower.x), 71)
        self.assertTrue(np.all(upper.pixels[:, 1] < 40))
        self.assertTrue(np.all(lower.pixels[:, 1] >= 45))
        points = {tuple(row) for row in all_pixels.pixels}
        self.assertEqual(points, {tuple(row) for cloud in (upper, lower) for row in cloud.pixels})
        self.assertFalse({tuple(row) for row in upper.pixels} & {tuple(row) for row in lower.pixels})
        self.assertIs(upper.calibration, all_pixels.calibration)
        for cloud in (upper, lower):
            np.testing.assert_allclose(cloud.calibration.x.transform(cloud.pixels[:, 0])[0], cloud.x)
            np.testing.assert_allclose(cloud.calibration.y.transform(cloud.pixels[:, 1])[1], cloud.eps[:, 1])

    def test_missing_curve_pixels_are_not_filled_when_the_boundary_is_interpolated(self):
        analysis = fixture()
        for x in range(40, 51):
            analysis[0][15 + (x - 20) // 5, x] = 1
        cloud = extract_color_data(analysis, "black")
        upper, lower = split_two_curves(cloud, min_separation_px=20)
        self.assertEqual(len(upper.x), 60)
        self.assertEqual(len(lower.x), 71)
        self.assertFalse(np.any((upper.pixels[:, 0] >= 40) & (upper.pixels[:, 0] <= 50)))

    def test_long_anchor_gap_is_reported_instead_of_silently_crossed(self):
        analysis = fixture()
        for x in range(30, 71):
            analysis[0][15 + (x - 20) // 5, x] = 1
        cloud = extract_color_data(analysis, "black")
        with self.assertRaises(ValueError):
            split_two_curves(cloud, min_separation_px=20, max_anchor_gap_px=20)

    def test_more_than_two_bands_or_only_one_curve_fail_explicitly(self):
        for count in (1, 3):
            analysis = fixture()
            analysis[0][:] = 1
            for i in range(count):
                analysis[0][15 + i * 20, 20:91] = 0
            with self.subTest(count=count), self.assertRaises(ValueError):
                split_two_curves(extract_color_data(analysis, "black"), min_separation_px=10)

    def test_exclusions_and_neutral_color_filter_remove_annotation_pixels(self):
        analysis = fixture()
        analysis[0][75, 50] = (0, 0, .2)
        excluded = Rectangle(20, 10, 30, 25)
        cloud = extract_color_data(analysis, "black", color_tolerance=1., max_channel_spread=.08,
                                   exclude_regions=(excluded,))
        self.assertNotIn((50, 75), {tuple(row) for row in cloud.pixels})
        self.assertFalse(any(excluded.left <= x <= excluded.right and excluded.top <= y <= excluded.bottom
                             for x, y in cloud.pixels))
        self.assertEqual(cloud.excluded_regions, (excluded,))

    def test_invalid_options_and_empty_cloud_fail(self):
        cloud = extract_color_data(fixture(), "black")
        for options in ({"min_separation_px": 0}, {"max_anchor_gap_px": np.inf}):
            with self.assertRaises(ValueError):
                split_two_curves(cloud, **options)
        analysis = fixture()
        analysis[0][:] = 1
        with self.assertRaises(ValueError):
            split_two_curves(extract_color_data(analysis, "black"))
        for options in ({"max_channel_spread": 2}, {"exclude_regions": [(1, 2, 3, 4)]}):
            with self.assertRaises(ValueError):
                extract_color_data(fixture(), "black", **options)

    def test_real_paper_clouds_use_original_calibration_and_keep_removed_pixels_out(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "real_example_2" / "real_example2.png"
        with tempfile.TemporaryDirectory() as directory:
            analysis = pre_analyzer(path, 3, output=Path(directory) / "precheck.png",
                                    x_ticks=np.r_[.009, np.arange(1, 10)*.01, .1, .2],
                                    y_ticks=np.r_[np.arange(4, 10)*.001, np.arange(1, 10)*.01,
                                                  np.arange(1, 10)*.1, 1, 2], x_log=True, y_log=True)
            removal = remove_colors(analysis[0], ["red", "blue"], hue_tolerance=18)
            exclusions = (Rectangle(1158, 74, 1626, 547), Rectangle(254, 832, 852, 1430))
            parent = extract_color_data((removal.image, *analysis[1:]), "black", color_tolerance=1.,
                                        frame_margin=35, max_channel_spread=.08, exclude_regions=exclusions)
            curves = split_two_curves(parent, min_separation_px=40, max_anchor_gap_px=80)
            self.assertEqual(sum(len(c.x) for c in curves), len(parent.x))
            self.assertEqual([c.calibration.x.scale for c in curves], ["log", "log"])
            self.assertEqual([c.calibration.y.scale for c in curves], ["log", "log"])
            for curve in curves:
                self.assertGreater(len(curve.x), 3500)
                self.assertGreater(np.ptp(curve.pixels[:, 0]), 1250)
                px, py = curve.pixels.T
                self.assertFalse(removal.mask[py, px].any())
                np.testing.assert_allclose(analysis[2].x_validation.pixel_slope*np.log(curve.x)
                                           + analysis[2].x_validation.pixel_offset, px)
                np.testing.assert_allclose(analysis[2].y_validation.pixel_slope*np.log(curve.y)
                                           + analysis[2].y_validation.pixel_offset, py)
            output = Path(directory) / "curves.png"
            self.assertEqual(plot_color_curves(analysis, curves, output=output), output)
            self.assertEqual(imread(output).shape[:2], (900, 1800))
