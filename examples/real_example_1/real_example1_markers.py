"""Find black marker hypotheses using the cropped reference from this paper."""

from pathlib import Path
import numpy as np

from figure_data_unwrapper import Rectangle, extract_markers, plot_marker_data
from real_example1_check import image, frame, ticks, colors

analysis = image, frame, ticks, colors
# This explicit region excludes the known legend dot. Its color/shape is
# identical to data markers, so template matching alone cannot classify it.
legend = Rectangle(200, 480, 320, 580)
cloud = extract_markers(analysis, Path(__file__).with_name("black_marker.png"),
                        scales=(1., 1.1, 1.2), exclude_regions=(legend,))
x, y, eps = cloud
output = Path(__file__).with_name("real_example1_markers")
np.savetxt(output.with_suffix(".csv"), cloud.to_array(), delimiter=",", comments="",
           header="x,y,eps_x,eps_y,pixel_x,pixel_y,scale,score,color_support,visible_fraction,mse,unique_gain,overlaps_neighbor")
plot_marker_data(analysis, cloud, output=output.with_suffix(".png"))
print(f"Fitted {len(x)} marker hypotheses from {cloud.candidate_count} candidates.")
print(f"Overlapping hypotheses: {sum(m.overlaps_neighbor for m in cloud.matches)}; "
      "eps contains calibration uncertainty only.")
