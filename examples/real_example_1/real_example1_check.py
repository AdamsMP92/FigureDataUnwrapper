"""Apply the preliminary analysis to a figure from a paper."""

from pathlib import Path
import numpy as np
from figure_data_unwrapper import pre_analyzer

n_colors = 7
file_path = Path(__file__).with_name("real_example1.png")
image, frame, ticks, colors = pre_analyzer(
    file_path, 
    n_colors, 
    x_ticks=np.linspace(0, 360, 17),
    y_ticks=[0, 2.5, 5, 7.5, 10], 
    x_lin=True, 
    y_lin=True,
)
