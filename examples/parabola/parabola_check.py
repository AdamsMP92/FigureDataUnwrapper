"""Analyze the parabola image and save its standard plausibility plot."""

from pathlib import Path
from figure_data_unwrapper import pre_analyzer

n_colors = 7
file_path = Path(__file__).with_name("parabola.png")
image, frame, ticks, colors = pre_analyzer(
    file_path, n_colors, x_ticks=range(-3, 4), y_ticks=range(0, 10, 2),
)
