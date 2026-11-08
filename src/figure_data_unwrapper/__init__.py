"""Tools for reconstructing data from scientific figures."""

from .analyzer import pre_analyzer
from .colors import rank_colors
from .frame import FrameDetection, detect_frame, fit_frame
from .plotting import plot_check
from .preprocessing import foreground_mask
from .rectangle import Rectangle, rectangle_model
from .ticks import (
    AxisTick, AxisTickValidation, GridLine, TickGridDetection, TickMark, TickOutlier,
    detect_ticks_and_grid,
)
from .tick_validation import check_tick_consistency

__all__ = [
    "AxisTick", "AxisTickValidation", "FrameDetection", "GridLine", "Rectangle",
    "TickGridDetection", "TickMark", "TickOutlier", "check_tick_consistency",
    "detect_frame", "detect_ticks_and_grid", "fit_frame",
    "foreground_mask", "plot_check", "pre_analyzer", "rank_colors", "rectangle_model",
]
