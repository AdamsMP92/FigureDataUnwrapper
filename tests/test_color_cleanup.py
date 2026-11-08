"""Approximate color removal preserves neutral strokes and pixel provenance."""

from pathlib import Path
import tempfile
import unittest

from matplotlib.image import imread
import numpy as np

from figure_data_unwrapper import plot_color_removal, remove_colors


class ColorRemovalTests(unittest.TestCase):
    def test_hue_selection_includes_light_dark_and_antialiased_red_and_blue(self):
        image = np.ones((10, 12, 3))
        image[4, 2:10] = [(1, 0, 0), (1, .8, .8), (.2, 0, 0), (1, 0, .04),
                          (0, 0, .6), (.8, .8, 1), (.4, .4, .4), (1, 1, 0)]
        removal = remove_colors(image, ["red", "blue"])
        np.testing.assert_array_equal(removal.mask[4, 2:10], [True] * 6 + [False] * 2)
        self.assertEqual(removal.removed_pixels, 6)
        self.assertEqual(removal.counts, (4, 2))
        np.testing.assert_array_equal(removal.image[removal.mask], np.ones((6, 3)))

    def test_neutral_grayscale_strokes_are_preserved_at_every_brightness(self):
        image = np.repeat(np.linspace(0, 1, 30)[None, :, None], 3, axis=2)
        image = np.repeat(image, 20, axis=0)
        removal = remove_colors(image, ["red", "blue"], min_saturation=.01)
        self.assertFalse(removal.mask.any())
        np.testing.assert_array_equal(removal.image, image)

    def test_source_is_unchanged_and_output_does_not_alias_it(self):
        image = np.ones((10, 12, 3))
        image[4, 4] = (1, 0, 0)
        source = image.copy()
        cleaned, mask = remove_colors(image, "red")
        self.assertTrue(mask[4, 4])
        cleaned[0, 0] = 0
        np.testing.assert_array_equal(image, source)

    def test_background_can_be_inferred_or_explicitly_supplied(self):
        image = np.full((10, 12, 3), .85)
        image[4, 4] = (1, 0, 0)
        inferred = remove_colors(image, "red")
        np.testing.assert_array_equal(inferred.image[4, 4], [.85] * 3)
        explicit = remove_colors(image, "red", background_color="#f0f0f0")
        np.testing.assert_allclose(explicit.image[4, 4], [240 / 255] * 3)
        np.testing.assert_array_equal(explicit.image[0, 0], [.85] * 3)

    def test_hue_tolerance_wraps_at_red_and_can_be_narrowed(self):
        image = np.ones((10, 12, 3))
        image[4, 3:6] = [(1, 0, .05), (1, .05, 0), (1, .5, 0)]
        removal = remove_colors(image, "red", hue_tolerance=5)
        np.testing.assert_array_equal(removal.mask[4, 3:6], [True, True, False])

    def test_saturation_and_background_contrast_are_independent_guards(self):
        image = np.ones((10, 12, 3))
        image[4, 4:6] = [(1, .98, .98), (1, .8, .8)]
        removal = remove_colors(image, "red", min_saturation=.01, min_contrast=.1)
        np.testing.assert_array_equal(removal.mask[4, 4:6], [False, True])

    def test_rgb_mode_supports_achromatic_targets(self):
        image = np.ones((10, 12, 3))
        image[4, 4:7] = [(0, 0, 0), (.1, .1, .1), (1, 0, 0)]
        removal = remove_colors(image, "black", mode="rgb", rgb_tolerance=.1)
        np.testing.assert_array_equal(removal.mask[4, 4:7], [True, False, False])
        np.testing.assert_array_equal(removal.image[4, 4], [1, 1, 1])

    def test_integer_rgba_input_is_composited_and_normalized(self):
        image = np.zeros((10, 12, 4), dtype=np.uint8)
        image[..., 0] = 255  # Invisible red should already be white after compositing.
        image[4, 4] = (0, 0, 255, 255)
        removal = remove_colors(image, "blue")
        self.assertEqual(removal.removed_pixels, 1)
        self.assertEqual(removal.image.shape, (10, 12, 3))
        np.testing.assert_array_equal(removal.image, np.ones((10, 12, 3)))

    def test_single_rgb_target_and_overlapping_targets_count_union_once(self):
        image = np.ones((10, 12, 3))
        image[4, 4] = (1, 0, 0)
        for target in ((255, 0, 0), np.array([1., 0., 0.])):
            self.assertEqual(remove_colors(image, target).removed_pixels, 1)
        removal = remove_colors(image, ["red", "#ef0800"])
        self.assertEqual(removal.counts, (1, 1))
        self.assertEqual(removal.removed_pixels, 1)

    def test_empty_targets_keep_an_independent_rgb_copy(self):
        image = np.full((10, 12), .5)
        removal = remove_colors(image, [])
        self.assertFalse(removal.mask.any())
        np.testing.assert_array_equal(removal.image, np.full((10, 12, 3), .5))

    def test_invalid_options_and_undefined_hues_are_rejected(self):
        for options in ({"mode": "unknown"}, {"hue_tolerance": 181}, {"min_saturation": 0},
                        {"rgb_tolerance": 2}, {"min_contrast": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                remove_colors(np.ones((10, 12, 3)), "red", **options)
        for target in ("black", "white", "gray", None, [256, 0, 0]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                remove_colors(np.ones((10, 12, 3)), target)
        with self.assertRaises(ValueError):
            remove_colors(np.array([[256]]), "red")

    def test_preview_exports_and_rejects_mismatched_geometry(self):
        image = np.ones((10, 12, 3))
        removal = remove_colors(image, "red")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cleanup.png"
            self.assertEqual(plot_color_removal(image, removal, output=path), path)
            self.assertEqual(imread(path).shape[:2], (900, 2025))
            with self.assertRaises(ValueError):
                plot_color_removal(np.ones((20, 20, 3)), removal, output=path)

    def test_real_paper_removes_colors_and_preserves_every_exact_neutral_pixel(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "real_example_2" / "real_example2.png"
        source = imread(path)
        original = source.copy()
        removal = remove_colors(source, ["red", "blue"], hue_tolerance=18)
        self.assertGreater(removal.removed_pixels, 100000)
        neutral = np.ptp(source[..., :3], axis=2) == 0
        self.assertFalse(removal.mask[neutral].any())
        np.testing.assert_array_equal(removal.image[neutral], source[..., :3][neutral])
        np.testing.assert_array_equal(removal.image[~removal.mask], source[..., :3][~removal.mask])
        np.testing.assert_array_equal(source, original)
