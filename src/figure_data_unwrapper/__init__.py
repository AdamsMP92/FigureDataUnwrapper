"""Tools for reconstructing data from scientific figures."""

from .frame import FrameDetection, detect_frame, fit_frame
from .preprocessing import foreground_mask
from .rectangle import Rectangle, rectangle_model

__all__ = [
    "FrameDetection", "Rectangle", "detect_frame", "fit_frame", "foreground_mask",
    "rectangle_model",
]
