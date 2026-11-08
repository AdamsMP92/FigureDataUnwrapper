"""Checks for exact RGB frequency counts and deterministic top-N ranking."""

import unittest

import numpy as np

from figure_data_unwrapper import rank_colors


class ColorRankingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.array([
            [[255, 255, 255], [255, 0, 0], [0, 0, 0]],
            [[255, 255, 255], [255, 0, 0], [255, 255, 255]],
        ], dtype=np.uint8)

    def test_ranking_and_top_n_limit(self):
        self.assertEqual(rank_colors(self.image, 2), [(255, 255, 255, 3), (255, 0, 0, 2)])

    def test_large_limit_returns_all_colors_and_counts_every_pixel(self):
        ranking = rank_colors(self.image, 20)
        self.assertEqual(ranking, [(255, 255, 255, 3), (255, 0, 0, 2), (0, 0, 0, 1)])
        self.assertEqual(sum(row[3] for row in ranking), 6)
        self.assertTrue(all(type(value) is int for row in ranking for value in row))

    def test_ties_are_ordered_by_rgb_not_image_position(self):
        image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])
        expected = [(0, 0, 255, 1), (0, 255, 0, 1), (255, 0, 0, 1)]
        self.assertEqual(rank_colors(image, 3), expected)
        self.assertEqual(rank_colors(image[:, ::-1], 3), expected)

    def test_float_channels_keep_normalized_scale(self):
        image = self.image.astype(float) / 255
        self.assertEqual(rank_colors(image, 3), [(1., 1., 1., 3), (1., 0., 0., 2), (0., 0., 0., 1)])
        self.assertIs(type(rank_colors(image, 1)[0][0]), float)

    def test_nearby_float_colors_are_not_quantized_or_merged(self):
        near = float(np.nextafter(0.5, 1))
        image = np.array([[[0.5, 0.5, 0.5], [near, 0.5, 0.5], [0.5, 0.5, 0.5]]])
        self.assertEqual(rank_colors(image, 2), [(0.5, 0.5, 0.5, 2), (near, 0.5, 0.5, 1)])

    def test_all_256_integer_levels_are_preserved(self):
        image = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
        self.assertEqual(rank_colors(image, 256), [(value, value, value, 1) for value in range(256)])

    def test_grayscale_is_reported_as_rgb(self):
        self.assertEqual(rank_colors(np.array([[128, 255], [128, 0]]), 3),
                         [(128, 128, 128, 2), (0, 0, 0, 1), (255, 255, 255, 1)])

    def test_integer_rgba_is_composited_over_white(self):
        image = np.array([[[0, 0, 0, 0], [255, 255, 255, 255], [255, 0, 0, 128]]], dtype=np.uint8)
        self.assertEqual(rank_colors(image, 3), [(255, 255, 255, 2), (255, 127, 127, 1)])

    def test_float_rgba_is_composited_without_eight_bit_quantization(self):
        image = np.array([[[0., 0., 0., 0.], [1., 0., 0., 0.5]]])
        self.assertEqual(rank_colors(image, 3), [(1., 0.5, 0.5, 1), (1., 1., 1., 1)])

    def test_zero_and_numpy_integer_limits(self):
        self.assertEqual(rank_colors(self.image, 0), [])
        self.assertEqual(rank_colors(self.image, np.int64(1)), [(255, 255, 255, 3)])

    def test_input_is_not_modified(self):
        before = self.image.copy()
        rank_colors(self.image, 2)
        np.testing.assert_array_equal(self.image, before)

    def test_invalid_limits(self):
        for count in (-1, 1.5, True, None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                rank_colors(self.image, count)

    def test_invalid_images(self):
        for image in (
            np.empty((0, 2, 3)), np.ones((2, 2, 2)), np.ones((2, 3, 4, 3)),
            np.array([[256]]), np.array([[-1]]), np.array([[1.1]]),
            np.array([[np.nan]]), np.array([[True]]), np.array([["red"]]),
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(ValueError):
                rank_colors(image, 3)
