"""Integration checks for the file-based preliminary analysis wrapper."""

from pathlib import Path
import tempfile
import unittest

from matplotlib.image import imread, imsave
import numpy as np

from figure_data_unwrapper import Rectangle, pre_analyzer


class PreAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input = Path(self.directory.name) / "figure.png"
        self.output = self.input.with_name("figure_check.png")
        image = np.ones((120, 160, 3))
        image[16:105, [50, 100]] = 0.8
        image[[40, 75], 21:140] = 0.8
        image[15, 20:141] = image[105, 20:141] = 0
        image[15:106, 20] = image[15:106, 140] = 0
        image[106:112, [50, 100]] = 0
        image[[40, 75], 14:20] = 0
        imsave(self.input, image)

    def test_full_pipeline_returns_results_and_saves_beside_input(self):
        original = self.input.read_bytes()
        image, frame, ticks, colors = pre_analyzer(str(self.input), n_colors=2)
        self.assertEqual(image.shape[:2], (120, 160))
        self.assertEqual(frame.rectangle, Rectangle(20, 15, 140, 105))
        self.assertIs(ticks.frame, frame)
        self.assertEqual([tick.position for tick in ticks.x_ticks], [50, 100])
        self.assertEqual([tick.position for tick in ticks.y_ticks], [40, 75])
        self.assertEqual(len(ticks.vertical_grid), 2)
        self.assertEqual(len(ticks.horizontal_grid), 2)
        self.assertEqual(len(colors), 2)
        self.assertGreater(colors[0][3], colors[1][3])
        self.assertEqual(imread(self.output).shape[1], 160 + 280)
        pixels = imread(self.output)[:120, :160, :3]
        orange = (pixels[..., 0] > 0.8) & (pixels[..., 1] > 0.3) & (pixels[..., 1] < 0.8) & (pixels[..., 2] < 0.2)
        self.assertTrue(orange.any())  # The wrapper enables tick plotting by default.
        self.assertEqual(self.input.read_bytes(), original)

    def test_custom_output_and_count_hints(self):
        output = Path(self.directory.name) / "checks" / "custom.png"
        _, _, ticks, colors = pre_analyzer(
            self.input, n_colors=7, output=output, expected_x_ticks=2,
            expected_y_ticks=3, show_ticks=True,
        )
        self.assertTrue(ticks.x_count_matches)
        self.assertFalse(ticks.y_count_matches)
        self.assertTrue(ticks.notes)
        self.assertLessEqual(len(colors), 7)
        self.assertTrue(output.is_file())
        self.assertFalse(self.output.exists())

    def test_failed_frame_still_produces_a_plausibility_plot(self):
        imsave(self.input, np.ones((120, 160, 3)))
        _, frame, ticks, colors = pre_analyzer(self.input, n_colors=7)
        self.assertIsNone(frame)
        self.assertIsNone(ticks)
        self.assertEqual(colors, [(1.0, 1.0, 1.0, 120 * 160)])
        self.assertTrue(self.output.is_file())

    def test_integer_jpeg_is_normalized_for_analysis(self):
        jpeg = self.input.with_suffix(".jpg")
        imsave(jpeg, np.full((20, 30, 3), 255, dtype=np.uint8))
        image, frame, ticks, colors = pre_analyzer(jpeg, n_colors=1)
        self.assertTrue(np.issubdtype(image.dtype, np.floating))
        np.testing.assert_array_equal(image, np.ones((20, 30, 3)))
        self.assertIsNone(frame)
        self.assertIsNone(ticks)
        self.assertEqual(colors, [(1.0, 1.0, 1.0, 600)])
        self.assertTrue(self.output.is_file())

    def test_input_image_cannot_be_overwritten(self):
        original = self.input.read_bytes()
        with self.assertRaises(ValueError):
            pre_analyzer(self.input, n_colors=7, output=self.input)
        self.assertEqual(self.input.read_bytes(), original)

    def test_manual_values_separate_real_paper_false_branches(self):
        paper = Path(__file__).resolve().parents[1] / "examples" / "real_example1.png"
        _, _, ticks, _ = pre_analyzer(
            paper, n_colors=7, output=self.output,
            x_ticks=np.linspace(0, 360, 17), y_ticks=[0, 2.5, 5, 7.5, 10],
            x_lin=True, y_lin=True,
        )
        self.assertEqual(len(ticks.x_ticks), 17)
        self.assertEqual(len(ticks.y_ticks), 5)
        self.assertEqual(len(ticks.rejected_y_ticks), 2)
        self.assertTrue(all(item.reason == "off_reference" for item in ticks.rejected_y_ticks))
        self.assertEqual([tick.value for tick in ticks.y_ticks], [10, 7.5, 5, 2.5, 0])
        pixels = imread(self.output)[:831, :1355, :3]
        purple = (pixels[..., 0] > 0.5) & (pixels[..., 1] < 0.3) & (pixels[..., 2] > 0.6)
        self.assertTrue(purple.any())

    def test_rgb_correction_precedes_grid_search_for_shadowed_paper(self):
        paper = Path(__file__).resolve().parents[1] / "examples" / "real_example2.png"
        _, frame, ticks, _ = pre_analyzer(paper, n_colors=3, output=self.output)
        self.assertTrue(frame.color_check.corrected)
        self.assertAlmostEqual(frame.rectangle.top, 27, delta=1)
        self.assertEqual(frame.color_check.status, "valid")
        self.assertIs(ticks.frame, frame)
        self.assertEqual(ticks.horizontal_grid, ())
        self.assertEqual(ticks.vertical_grid, ())
