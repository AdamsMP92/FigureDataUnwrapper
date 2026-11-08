"""Extract the connected red curve, reusing the paper's existing pre-analysis."""

from pathlib import Path
import numpy as np

from figure_data_unwrapper import extract_color_data, plot_color_data
from real_example1_check import image, frame, ticks, colors

analysis = image, frame, ticks, colors

cloud = extract_color_data(analysis, "#EF0800", components="largest", color_tolerance=0.1)

output = Path(__file__).with_name("real_example1_red")
np.savetxt(output.with_suffix(".csv"), cloud.to_array(), delimiter=",",
           header="x,y,eps_x,eps_y,pixel_x,pixel_y", comments="")
plot_color_data(analysis, cloud, output=output.with_suffix(".png"))

