"""Analyze a log-log paper figure with inset images and major/minor ticks."""

from pathlib import Path
import numpy as np
from figure_data_unwrapper import pre_analyzer

n_colors = 7
file_path = Path(__file__).with_name("real_example2.png")
# Major/minor values, x left-to-right and y bottom-to-top. Corner marks
# coincident with a frame stroke are not independently visible inward ticks.
x_ticks = np.r_[0.009, np.arange(1, 10) * 0.01, 0.1, 0.2]
y_ticks = np.r_[np.arange(4, 10) * 0.001, np.arange(1, 10) * 0.01,
                np.arange(1, 10) * 0.1, 1, 2]

image, frame, ticks, colors = pre_analyzer(
    file_path, 
    n_colors, 
    x_ticks=x_ticks, 
    y_ticks=y_ticks,
    x_log=True, 
    y_log=True,
)
