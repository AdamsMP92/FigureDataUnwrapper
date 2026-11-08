"""Run the preliminary image analysis and save its standard plausibility plot."""

from pathlib import Path

from matplotlib.image import imread
import numpy as np
from numpy.typing import ArrayLike

from .colors import ColorCount, rank_colors
from .frame import FrameDetection, fit_frame
from .plotting import plot_check
from .preprocessing import foreground_mask
from .ticks import TickGridDetection, detect_ticks_and_grid
from .tick_validation import check_tick_consistency


def pre_analyzer(
    file_path: str | Path,
    n_colors: int,
    *,
    output: str | Path | None = None,
    expected_x_ticks: int | None = None,
    expected_y_ticks: int | None = None,
    show_ticks: bool = True,
    x_ticks: ArrayLike | None = None,
    y_ticks: ArrayLike | None = None,
    x_lin: bool = False,
    y_lin: bool = False,
    x_log: bool = False,
    y_log: bool = False,
    tick_tolerance_px: float = 2.0,
) -> tuple[np.ndarray, FrameDetection | None, TickGridDetection | None, list[ColorCount]]:
    """Read a figure, analyze its geometry/colors, and write a PNG check plot.

    Return (image, frame, ticks, colors), with image normalized to [0, 1].
    Integer rasters such as JPEGs are normalized before extraction. PNG floats
    retain their original values. A failed frame fit returns None for frame and
    ticks while still ranking colors and saving the image with a status note.

    By default the check plot is saved beside the input as <stem>_check.png.
    output overrides its location; the source image cannot be overwritten.
    Optional tick counts and tick visibility are forwarded to the respective
    analysis and plotting functions. Detected ticks are plotted by default;
    set show_ticks=False to hide them.

    x_ticks/y_ticks optionally supply numeric axis values, ordered left-to-right
    for x and bottom-to-top for y. Select *_lin or *_log per axis; both flags
    together are invalid. Manual values default to linear. Consistency checks
    separate unsupported observations into ticks.rejected_x_ticks/y_ticks and
    attach validation details without changing this function's return tuple.
    """
    input_path = Path(file_path)
    output_path = Path(output) if output is not None else input_path.with_name(
        f"{input_path.stem}_check.png"
    )
    if output_path.resolve() == input_path.resolve():
        raise ValueError("The check plot output must differ from the input image.")

    image = imread(input_path)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float64) / np.iinfo(image.dtype).max
    frame = fit_frame(foreground_mask(image))
    ticks = detect_ticks_and_grid(
        image, frame=frame,
        expected_x_ticks=None if x_ticks is not None or x_lin or x_log else expected_x_ticks,
        expected_y_ticks=None if y_ticks is not None or y_lin or y_log else expected_y_ticks,
    ) if frame is not None else None
    ticks = check_tick_consistency(
        ticks, x_ticks=x_ticks, y_ticks=y_ticks,
        x_lin=x_lin, y_lin=y_lin, x_log=x_log, y_log=y_log,
        tolerance_px=tick_tolerance_px,
        expected_x_ticks=expected_x_ticks, expected_y_ticks=expected_y_ticks,
    )
    colors = rank_colors(image, n_colors=n_colors)
    plot_check(image, frame, ticks, colors, output=output_path, show_ticks=show_ticks)
    return image, frame, ticks, colors
