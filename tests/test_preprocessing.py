"""Checks for color-preserving foreground/background binarization."""

import unittest

import numpy as np

from figure_data_unwrapper import foreground_mask


class ForegroundMaskTests(unittest.TestCase):
    def test_contrast_is_relative_to_light_or_dark_background(self):
        for background, stroke in ((1.0, 0.93), (0.0, 0.07), (0.5, 0.7)):
            with self.subTest(background=background, stroke=stroke):
                image = np.full((20, 30), background)
                image[8, 5:25] = stroke
                expected = np.zeros(image.shape, dtype=bool)
                expected[8, 5:25] = True
                np.testing.assert_array_equal(foreground_mask(image), expected)

    def test_rgb_preserves_a_color_with_almost_no_luminance_contrast(self):
        image = np.full((20, 30, 3), 0.5)
        image[8, 5:25] = [0.7, 0.44, 0.5]
        self.assertTrue(foreground_mask(image)[8, 15])
        grayscale = image @ np.array([0.2126, 0.7152, 0.0722])
        self.assertFalse(foreground_mask(grayscale).any())

    def test_small_background_variations_are_suppressed(self):
        rng = np.random.default_rng(9)
        image = rng.uniform(0.985, 1, size=(40, 60))
        image[15, 10:50] = 0.9
        expected = np.zeros(image.shape, dtype=bool)
        expected[15, 10:50] = True
        np.testing.assert_array_equal(foreground_mask(image), expected)

    def test_transparency_is_composited_over_white(self):
        image = np.zeros((20, 30, 4))
        image[8, 5:25] = [1, 1, 0, 1]
        mask = foreground_mask(image)
        self.assertEqual(np.count_nonzero(mask), 20)
        image[..., 3] = 0
        self.assertFalse(foreground_mask(image).any())

    def test_boolean_foreground_is_preserved_without_aliasing(self):
        source = np.zeros((20, 30), dtype=bool)
        source[8, 5:25] = True
        mask = foreground_mask(source)
        np.testing.assert_array_equal(source, mask)
        mask[8, 5] = False
        self.assertTrue(source[8, 5])

    def test_invalid_inputs_are_rejected(self):
        for image in (np.empty((0, 4)), np.ones((4, 4, 2)), [[np.inf]], [[255]]):
            with self.subTest(shape=np.shape(image)), self.assertRaises(ValueError):
                foreground_mask(image)
        for threshold in (0, 1, np.nan):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                foreground_mask(np.ones((20, 30)), min_contrast=threshold)
