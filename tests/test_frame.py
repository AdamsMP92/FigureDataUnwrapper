"""Behavioral checks using independently drawn synthetic images."""

import unittest

import numpy as np

from figure_data_unwrapper import Rectangle, detect_frame, rectangle_model


def draw_frame(image, left, top, right, bottom, *, thickness=1, value=0.0):
    """Draw inclusive edges by array slicing, independently of rectangle_model."""
    radius = thickness // 2
    image[top - radius:top + radius + 1, left:right + 1] = value
    image[bottom - radius:bottom + radius + 1, left:right + 1] = value
    image[top:bottom + 1, left - radius:left + radius + 1] = value
    image[top:bottom + 1, right - radius:right + radius + 1] = value


class RectangleTests(unittest.TestCase):
    def test_model_draws_outline_with_empty_interior(self):
        rectangle = Rectangle(4, 3, 12, 9)
        expected = np.ones((12, 16))
        draw_frame(expected, 4, 3, 12, 9)
        np.testing.assert_array_equal(rectangle_model(expected.shape, rectangle), expected == 0)
        self.assertEqual((rectangle.width, rectangle.height), (8, 6))

    def test_half_pixel_model_and_clipping(self):
        mask = rectangle_model((10, 10), Rectangle(-2, 2.5, 7.5, 12))
        self.assertTrue(mask[2, 4] and mask[3, 4])
        self.assertTrue(mask[6, 7] and mask[6, 8])
        self.assertFalse(mask[6, 4])

    def test_invalid_geometry_and_model_parameters(self):
        for bounds in [(2, 1, 2, 5), (2, 5, 8, 1), (0, 0, np.inf, 5)]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                Rectangle(*bounds)
        for shape, width in [((0, 10), 1), ((10.5, 20), 1), ((10, 20), 0)]:
            with self.subTest(shape=shape, width=width), self.assertRaises(ValueError):
                rectangle_model(shape, Rectangle(1, 1, 5, 5), line_width=width)


class FrameDetectionTests(unittest.TestCase):
    def test_frame_with_grid_points_and_tick_marks(self):
        image = np.ones((100, 140))
        image[20:81:10, 15:126] = 0.8
        image[10:91, 25:116:15] = 0.8
        image[94:97, 20:40] = 0  # Label-like distractor outside the plot.
        image[30:34, 50:54] = 0.2  # Data marker.
        draw_frame(image, 15, 10, 125, 90)
        image[90:95, 40:110:20] = 0  # Ticks extending beyond the bottom.
        self.assertEqual(detect_frame(image), Rectangle(15, 10, 125, 90))

    def test_largest_rectangle_wins_over_dark_grid_cells(self):
        image = np.ones((100, 140))
        image[30:71:20, 15:126] = 0
        image[10:91, 45:106:30] = 0
        draw_frame(image, 15, 10, 125, 90)
        self.assertEqual(detect_frame(image), Rectangle(15, 10, 125, 90))

    def test_varying_size_position_thickness_and_small_gaps(self):
        rng = np.random.default_rng(5)
        for thickness in (1, 3, 5):
            for _ in range(4):
                left, top = rng.integers(10, 30, size=2)
                right, bottom = rng.integers(85, 115, size=2)
                with self.subTest(bounds=(left, top, right, bottom), thickness=thickness):
                    image = np.ones((130, 140))
                    draw_frame(image, left, top, right, bottom, thickness=thickness)
                    image[top - 3:top + 4, left + 20:left + 23] = 1
                    self.assertEqual(detect_frame(image), Rectangle(left, top, right, bottom))

    def test_missing_top_is_inferred_from_remaining_endpoints(self):
        image = np.ones((100, 140))
        draw_frame(image, 15, 10, 125, 90)
        image[10, 16:125] = 1  # Preserve the two vertical segment endpoints.
        self.assertEqual(detect_frame(image), Rectangle(15, 10, 125, 90))

    def test_blank_and_solid_images_have_no_frame(self):
        for image in (np.ones((100, 140)), np.zeros((100, 140))):
            with self.subTest(value=image[0, 0]):
                self.assertIsNone(detect_frame(image))

    def test_invalid_image_and_parameters(self):
        for image in (np.empty((0, 2)), np.ones((4, 4, 2)), [[np.nan]], [[255]]):
            with self.subTest(shape=np.shape(image)), self.assertRaises(ValueError):
                detect_frame(image)
        for kwargs in ({"min_coverage": 0}, {"darkness_threshold": 2}, {"max_candidates": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detect_frame(np.ones((10, 10)), **kwargs)

    def test_matplotlib_frame_matches_renderer_geometry(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots()
        axes.scatter([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4])
        axes.grid(alpha=0.3)
        canvas.draw()

        rgb = np.asarray(canvas.buffer_rgba())[..., :3] / 255
        frame = detect_frame(rgb @ np.array([0.2126, 0.7152, 0.0722]))
        self.assertIsNotNone(frame)
        bounds = axes.get_window_extent()
        height = rgb.shape[0]
        # Display coordinates use the bottom-left image corner as origin;
        # array indices use the top-left pixel center. Allow raster snapping.
        expected = [
            bounds.x0 - 0.5, height - bounds.y1 - 0.5,
            bounds.x1 - 0.5, height - bounds.y0 - 0.5,
        ]
        np.testing.assert_allclose(
            [frame.left, frame.top, frame.right, frame.bottom],
            expected, rtol=0, atol=1,
        )


if __name__ == "__main__":
    unittest.main()
