"""Replace red/blue image content by the page background and retain its mask."""

from pathlib import Path
from matplotlib.image import imread, imsave

from figure_data_unwrapper import plot_color_removal, remove_colors

directory = Path(__file__).parent
image = imread(directory / "real_example2.png")
removal = remove_colors(image, ["red", "blue"], hue_tolerance=18, min_saturation=.1)
imsave(directory / "real_example2_cleaned.png", removal.image)
imsave(directory / "real_example2_removed_mask.png", removal.mask, cmap="gray", vmin=0, vmax=1)
plot_color_removal(image, removal, output=directory / "real_example2_color_removal.png")
print(f"Removed {removal.removed_pixels:,} pixels; background RGB = {removal.background_color}.")
