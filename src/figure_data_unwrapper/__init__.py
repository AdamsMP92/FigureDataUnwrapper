"""Tools for reconstructing data from scientific figures."""

from .analyzer import pre_analyzer
from .colors import rank_colors
from .color_cleanup import ColorRemoval, remove_colors
from .frame import FrameColorChange, FrameColorValidation, FrameDetection, FrameEdgeColor, detect_frame, fit_frame
from .frame_colors import check_frame_colors
from .extraction import AxisCalibration, ColorPointCloud, CoordinateCalibration, calibrate_axes, extract_color_data, split_two_curves
from .markers import MarkerMatch, MarkerPointCloud, extract_markers
from .plotting import plot_check, plot_color_curves, plot_color_data, plot_color_removal, plot_marker_data
from .preprocessing import foreground_mask
from .rectangle import Rectangle, rectangle_model
from .ticks import (
    AxisTick, AxisTickValidation, GridLine, TickGridDetection, TickMark, TickOutlier,
    detect_ticks_and_grid,
)
from .tick_validation import check_tick_consistency

__all__ = [
    "AxisCalibration", "AxisTick", "AxisTickValidation", "ColorPointCloud", "ColorRemoval", "CoordinateCalibration",
    "FrameColorChange", "FrameColorValidation",
    "FrameDetection", "FrameEdgeColor", "GridLine", "MarkerMatch", "MarkerPointCloud", "Rectangle",
    "TickGridDetection", "TickMark", "TickOutlier", "check_frame_colors", "check_tick_consistency",
    "calibrate_axes", "detect_frame", "detect_ticks_and_grid", "extract_color_data", "extract_markers", "fit_frame",
    "foreground_mask", "plot_check", "plot_color_curves", "plot_color_data", "plot_color_removal", "plot_marker_data",
    "pre_analyzer", "rank_colors", "rectangle_model", "remove_colors", "split_two_curves",
]
