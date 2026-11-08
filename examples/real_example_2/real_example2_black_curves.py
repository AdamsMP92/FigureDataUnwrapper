"""Separate the two visible black curves, retaining gaps and source pixels."""

from pathlib import Path
import numpy as np

from figure_data_unwrapper import Rectangle, extract_color_data, plot_color_curves, remove_colors, split_two_curves
from real_example2_check import image, frame, ticks, colors

removal = remove_colors(image, ["red", "blue"], hue_tolerance=18)
# Insets and inward ticks are explicitly excluded; they are not curve pixels.
insets = (Rectangle(1158, 74, 1626, 547), Rectangle(254, 832, 852, 1430))
pixels = extract_color_data((removal.image, frame, ticks, colors), "black",
                            color_tolerance=1., max_channel_spread=.08,
                            frame_margin=35, exclude_regions=insets)
upper, lower = split_two_curves(pixels, min_separation_px=40, max_anchor_gap_px=80)
directory = Path(__file__).parent
for name, cloud in (("upper", upper), ("lower", lower)):
    np.savetxt(directory / f"real_example2_black_{name}.csv", cloud.to_array(), delimiter=",", comments="",
               header="x,y,eps_x,eps_y,pixel_x,pixel_y")
    print(f"{name}: {len(cloud.x)} original pixels; calibration = log-log.")
plot_color_curves((image, frame, ticks, colors), (upper, lower),
                  output=directory / "real_example2_black_curves.png")
