"""Extract the parabola's markers using the cropped blue reference PNG."""

from pathlib import Path
import numpy as np

from figure_data_unwrapper import extract_markers, plot_marker_data
from parabola_check import image, frame, ticks, colors

analysis = image, frame, ticks, colors
cloud = extract_markers(analysis, Path(__file__).with_name("blue_marker.png"))
x, y, eps = cloud
output = Path(__file__).with_name("parabola_markers")
np.savetxt(output.with_suffix(".csv"), cloud.to_array(), delimiter=",", comments="",
           header="x,y,eps_x,eps_y,pixel_x,pixel_y,scale,score,color_support,visible_fraction,mse,unique_gain,overlaps_neighbor")
plot_marker_data(analysis, cloud, output=output.with_suffix(".png"))
print(f"Extracted {len(x)} blue markers; eps contains calibration uncertainty only.")
