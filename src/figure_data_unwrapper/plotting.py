"""Reusable PNG plausibility plots in the original image's pixel coordinates."""

from collections.abc import Sequence
from pathlib import Path
import textwrap

import numpy as np
from numpy.typing import ArrayLike

from .colors import ColorCount
from .color_cleanup import ColorRemoval
from .extraction import ColorPointCloud, _analysis_parts
from .markers import MarkerPointCloud
from .frame import FrameDetection
from .preprocessing import _normalized_image
from .ticks import TickGridDetection


def _display_rgb(image: ArrayLike):
    source = np.asarray(image)
    if source.dtype == np.bool_ and source.ndim == 2:
        source = (~source).astype(float)
    elif np.issubdtype(source.dtype, np.integer):
        if source.size == 0 or source.min() < 0 or source.max() > 255:
            raise ValueError("Integer image channels must be in [0, 255].")
        source = source.astype(float) / 255
    rgb = _normalized_image(source)
    return np.repeat(rgb[..., None], 3, axis=2) if rgb.ndim == 2 else rgb


def _swatch(row: ColorCount):
    channels = np.asarray(row[:3], dtype=float)
    scale = 255 if all(isinstance(value, (int, np.integer)) for value in row[:3]) else 1
    if channels.shape != (3,) or not np.isfinite(channels).all() or (
        (channels < 0).any() or (channels > scale).any()
    ):
        raise ValueError("Ranking entries need RGB in [0, 255] (integer) or [0, 1] (float).")
    if len(row) != 4 or isinstance(row[3], (bool, np.bool_)) or not isinstance(
        row[3], (int, np.integer)
    ) or row[3] < 0:
        raise ValueError("Ranking entries need a nonnegative integer pixel count.")
    rgb = channels / scale
    label = "#" + "".join(f"{round(value * 255):02X}" for value in rgb)
    return rgb, label, int(row[3])


def _draw_ticks(axes, rectangle, ticks, *, color, rejected=False):
    for tick in ticks:
        for mark in tick.marks:
            x, y = (mark.position, getattr(rectangle, mark.side)) if mark.side in (
                "top", "bottom"
            ) else (getattr(rectangle, mark.side), mark.position)
            direction = 1 if mark.side in ("top", "left") else -1
            normal = getattr(rectangle, mark.side) + direction * np.array([
                -mark.outward_length, mark.inward_length,
            ])
            tick_x, tick_y = ([x, x], normal) if mark.side in (
                "top", "bottom"
            ) else (normal, [y, y])
            axes.plot(tick_x, tick_y, color=color, linewidth=1.5,
                      linestyle=":" if rejected else "-", snap=False, zorder=4)
            marker_options = {"color": color} if rejected else {"facecolors": "none", "edgecolors": color}
            axes.scatter(x, y, s=48 if rejected else 32, marker="x" if rejected else "o",
                         linewidths=1.2, zorder=5, **marker_options)


def plot_check(
    image: ArrayLike,
    frame: FrameDetection | None = None,
    ticks: TickGridDetection | None = None,
    colors: Sequence[ColorCount] | None = None,
    *,
    output: str | Path = "check.png",
    show_ticks: bool = True,
) -> Path:
    """Save original pixels with a frame/grid overlay and an information panel.

    Observed frame sides are solid red, inferred sides dashed red, and grid
    candidates dashed cyan. Detected ticks are orange segments with circles at
    the frame intersection, enabled by default; show_ticks=False hides them.
    Separated tick candidates are dotted purple with crosses; their evidence
    remains visible for review rather than silently disappearing.
    The source image
    occupies the upper-left region at its native pixel size; the information
    panel adds space to the right and may extend the canvas height.

    No analysis is run here. If frame is omitted, use ticks.frame when available;
    if both are absent, save the original and report that no frame is available.
    Explicit frame/tick results must describe the same rectangle and lie inside
    the image. Up to ten supplied ranking entries are shown; exact RGB values
    remain in the original ranking, while panel swatches use hex RGB labels.

    Accept normalized grayscale/RGB/RGBA or integer channels in [0, 255]. RGBA
    is composited over white. Boolean masks are black foreground on white.
    Always write PNG without opening a GUI, return its path, and release the
    temporary figure. Numeric tick values are not inferred by this function.
    """
    rgb = _display_rgb(image)
    height, width = rgb.shape[:2]
    if ticks is not None and not isinstance(ticks, TickGridDetection):
        raise TypeError("ticks must be a TickGridDetection or None.")
    if frame is None and ticks is not None:
        frame = ticks.frame
    if frame is not None and not isinstance(frame, FrameDetection):
        raise TypeError("frame must be a FrameDetection or None.")
    if ticks is not None and frame.rectangle != ticks.frame.rectangle:
        raise ValueError("Frame and tick/grid results describe different rectangles.")
    if frame is not None:
        r = frame.rectangle
        if not (0 <= r.left < r.right < width and 0 <= r.top < r.bottom < height):
            raise ValueError("The frame lies outside this image.")
    path = Path(output)
    if path.suffix.lower() != ".png":
        raise ValueError("The check plot output must have a .png extension.")
    swatches = [_swatch(row) for row in colors[:10]] if colors is not None else []
    notes = list(ticks.notes) if ticks is not None else []
    if frame is not None and frame.color_check is not None:
        notes = [*frame.color_check.notes, *notes]
    if frame is None:
        notes.insert(0, "No frame result available.")
    elif ticks is None:
        notes.insert(0, "Tick/grid analysis not provided.")
    note_lines = [line for note in notes for line in textwrap.wrap(note, width=32)]
    if len(note_lines) > 5:
        note_lines = note_lines[:4] + ["Further notes in analysis result."]
    has_rejected = ticks is not None and bool(ticks.rejected_x_ticks or ticks.rejected_y_ticks)
    extra = 42 if has_rejected else 0

    # Import rendering only when requested; no pyplot state or GUI backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle as RectanglePatch

    panel_width = 280
    total_width = width + panel_width
    total_height = max(height, 222 + extra + 22 * len(swatches) + 16 * len(note_lines))
    dpi = 100
    figure = Figure(figsize=(total_width / dpi, total_height / dpi), dpi=dpi, facecolor="white")
    FigureCanvasAgg(figure)
    try:
        axes = figure.add_axes((0, 1 - height / total_height, width / total_width, height / total_height))
        axes.imshow(rgb, origin="upper", interpolation="nearest")
        axes.set_xlim(-0.5, width - 0.5)
        axes.set_ylim(height - 0.5, -0.5)
        axes.set_axis_off()
        if frame is not None:
            r = frame.rectangle
            edges = {
                "left": ([r.left, r.left], [r.top, r.bottom]),
                "top": ([r.left, r.right], [r.top, r.top]),
                "right": ([r.right, r.right], [r.top, r.bottom]),
                "bottom": ([r.left, r.right], [r.bottom, r.bottom]),
            }
            for side, (x, y) in edges.items():
                axes.plot(x, y, color="#e63946", linewidth=1,
                          linestyle="-" if side in frame.visible_sides else "--", snap=False)
        if ticks is not None:
            for line in (*ticks.vertical_grid, *ticks.horizontal_grid):
                x, y = ([line.position, line.position], [line.start, line.end]) if (
                    line.orientation == "vertical"
                ) else ([line.start, line.end], [line.position, line.position])
                axes.plot(x, y, color="#00a6b2", linewidth=1, linestyle="--", snap=False)
            if show_ticks:
                _draw_ticks(axes, r, (*ticks.x_ticks, *ticks.y_ticks), color="#ff9500")
                _draw_ticks(axes, r,
                            (item.tick for item in (*ticks.rejected_x_ticks, *ticks.rejected_y_ticks)),
                            color="#b025d3", rejected=True)

        panel = figure.add_axes(((width + 16) / total_width, 0, (panel_width - 32) / total_width, 1))
        panel.set(xlim=(0, panel_width - 32), ylim=(total_height, 0))
        panel.set_axis_off()
        panel.text(0, 24, "Plausibility check", fontsize=12, weight="bold")
        for y, label, color, style in (
            (50, "Frame: observed", "#e63946", "-"),
            (72, "Frame: inferred", "#e63946", "--"),
            (94, "Grid candidates", "#00a6b2", "--"),
        ):
            panel.plot([0, 24], [y, y], color=color, linestyle=style, linewidth=1)
            panel.text(32, y + 4, label, fontsize=9)
        if show_ticks:
            panel.scatter(12, 116, s=32, facecolors="none", edgecolors="#ff9500")
            panel.text(32, 120, "Tick marks", fontsize=9)
            if has_rejected:
                panel.scatter(12, 138, s=48, marker="x", color="#b025d3")
                panel.text(32, 142, "Separated ticks", fontsize=9)
        if ticks is not None:
            panel.text(0, 146 + extra / 2, f"Ticks: x = {len(ticks.x_ticks)}, y = {len(ticks.y_ticks)}", fontsize=9)
            if has_rejected:
                panel.text(0, 188, f"Separated: x = {len(ticks.rejected_x_ticks)}, y = {len(ticks.rejected_y_ticks)}", fontsize=9)
            panel.text(0, 166 + extra, f"Grid: {len(ticks.vertical_grid)} vertical, {len(ticks.horizontal_grid)} horizontal", fontsize=9)
        if frame is not None and frame.color_check is not None:
            color_status = frame.color_check.status
            if frame.color_check.corrected:
                color_status += " (corrected)"
            panel.text(0, 186 + extra, f"Frame RGB: {color_status}", fontsize=8)
        if swatches:
            panel.text(0, 200 + extra, f"Top colors ({len(swatches)} of {len(colors)})", fontsize=10, weight="bold")
            for index, (color, label, count) in enumerate(swatches):
                y = 214 + extra + 22 * index
                panel.add_patch(RectanglePatch((0, y), 20, 14, facecolor=color,
                                               edgecolor="#888888", linewidth=0.5))
                panel.text(30, y + 12, f"{label}  {count:,} px", fontsize=9)
        for index, line in enumerate(note_lines):
            panel.text(0, 216 + extra + 22 * len(swatches) + 16 * index, line, fontsize=8)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="png", dpi=dpi)
    finally:
        figure.clear()
    return path


def plot_color_data(analysis, cloud: ColorPointCloud, *, output: str | Path = "color_data.png") -> Path:
    """Show selected source pixels beside the calibrated, unsmoothed cloud.

    Sparse error bars show the axis-wise 1-sigma calibration uncertainty, not
    experimental errors or the rendered stroke width. No analysis is rerun.
    """
    rgb, frame, _ = _analysis_parts(analysis)
    if not isinstance(cloud, ColorPointCloud):
        raise TypeError("cloud must be a ColorPointCloud.")
    if cloud.image_shape != rgb.shape[:2] or cloud.frame.rectangle != frame.rectangle:
        raise ValueError("Cloud and analysis describe different image geometry.")
    path = Path(output)
    if path.suffix.lower() != ".png":
        raise ValueError("The extraction plot output must have a .png extension.")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 5.3), dpi=150, layout="constrained")
    FigureCanvasAgg(figure)
    try:
        source, calibrated = figure.subplots(1, 2)
        source.imshow(rgb, origin="upper", interpolation="nearest")
        source.scatter(cloud.pixels[:, 0], cloud.pixels[:, 1], s=0.3, c="#00a6b2", linewidths=0)
        source.set_title(f"Selected pixels: {len(cloud.x):,} / {cloud.matched_pixel_count:,} color matches")
        source.set_axis_off()
        calibrated.scatter(cloud.x, cloud.y, s=0.4, color=cloud.color, linewidths=0)
        if len(cloud.x):
            sample = np.unique(np.linspace(0, len(cloud.x) - 1, min(40, len(cloud.x))).astype(int))
            calibrated.errorbar(cloud.x[sample], cloud.y[sample], xerr=cloud.eps[sample, 0],
                                yerr=cloud.eps[sample, 1], fmt="none", ecolor="#333333",
                                elinewidth=0.5, label="Calibration 1 sigma (sampled)")
            calibrated.legend(fontsize=8)
        r = frame.rectangle
        if cloud.calibration.x.scale == "log":
            calibrated.set_xscale("log")
        if cloud.calibration.y.scale == "log":
            calibrated.set_yscale("log")
        calibrated.set_xlim(cloud.calibration.x.transform([r.left, r.right])[0])
        calibrated.set_ylim(cloud.calibration.y.transform([r.bottom, r.top])[0])
        calibrated.set(title="Calibrated pixel cloud", xlabel="x (axis units)", ylabel="y (axis units)")
        calibrated.grid(alpha=0.2)
        figure.suptitle("Color extraction; calibration uncertainty excludes stroke width and experimental errors", fontsize=10)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="png")
    finally:
        figure.clear()
    return path


def plot_marker_data(analysis, cloud: MarkerPointCloud, *, output: str | Path = "markers.png") -> Path:
    """Overlay numbered marker hypotheses and plot their calibrated centers."""
    rgb, frame, _ = _analysis_parts(analysis)
    if not isinstance(cloud, MarkerPointCloud):
        raise TypeError("cloud must be a MarkerPointCloud.")
    if cloud.image_shape != rgb.shape[:2] or cloud.frame.rectangle != frame.rectangle:
        raise ValueError("Markers and analysis describe different image geometry.")
    path = Path(output)
    if path.suffix.lower() != ".png":
        raise ValueError("The marker plot output must have a .png extension.")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle as RectanglePatch

    figure = Figure(figsize=(13, 5.5), dpi=150, layout="constrained")
    FigureCanvasAgg(figure)
    try:
        source, calibrated = figure.subplots(1, 2)
        source.imshow(rgb, origin="upper", interpolation="nearest")
        for index, match in enumerate(cloud.matches):
            source.scatter(match.pixel_x, match.pixel_y, s=65, facecolors="none", edgecolors="#00a6b2", linewidths=.8)
            source.text(match.pixel_x, match.pixel_y - 20, str(index), fontsize=5, color="#087177")
        for region in cloud.excluded_regions:
            source.add_patch(RectanglePatch((region.left, region.top), region.width, region.height,
                                            fill=False, edgecolor="#da9100", linestyle="--", linewidth=1))
        source.set(title=f"Marker hypotheses: {len(cloud.matches)}; numbered centers")
        source.set_axis_off()
        calibrated.errorbar(cloud.x, cloud.y, xerr=cloud.eps[:, 0], yerr=cloud.eps[:, 1],
                            fmt="o", markersize=3, color=cloud.color, elinewidth=.6)
        if cloud.calibration.x.scale == "log":
            calibrated.set_xscale("log")
        if cloud.calibration.y.scale == "log":
            calibrated.set_yscale("log")
        r = frame.rectangle
        calibrated.set_xlim(cloud.calibration.x.transform([r.left, r.right])[0])
        calibrated.set_ylim(cloud.calibration.y.transform([r.bottom, r.top])[0])
        calibrated.set(title="Calibrated marker centers", xlabel="x (axis units)", ylabel="y (axis units)")
        calibrated.grid(alpha=.2)
        figure.suptitle("Reference-template hypotheses; error bars show calibration uncertainty only", fontsize=10)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="png")
    finally:
        figure.clear()
    return path


def plot_color_removal(image: ArrayLike, removal: ColorRemoval, *, output: str | Path = "color_removal.png") -> Path:
    """Compare original, cleaned copy and the exact removed-pixel mask."""
    rgb = _display_rgb(image)
    if not isinstance(removal, ColorRemoval):
        raise TypeError("removal must be a ColorRemoval result.")
    if rgb.shape != removal.image.shape or removal.mask.shape != rgb.shape[:2]:
        raise ValueError("Original and color-removal result describe different image geometry.")
    path = Path(output)
    if path.suffix.lower() != ".png":
        raise ValueError("The color-removal plot output must have a .png extension.")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(13.5, 6), dpi=150, layout="constrained")
    FigureCanvasAgg(figure)
    try:
        original, cleaned, mask = figure.subplots(1, 3)
        original.imshow(rgb, origin="upper", interpolation="nearest")
        cleaned.imshow(removal.image, origin="upper", interpolation="nearest")
        mask.imshow(removal.mask, origin="upper", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for axes, title in ((original, "Original"), (cleaned, "Selected colors replaced by background"),
                            (mask, f"White = removed: {removal.removed_pixels:,} pixels")):
            axes.set_title(title, fontsize=10)
            axes.set_axis_off()
        figure.suptitle("Removed regions remain unknown; covered curve segments are not reconstructed", fontsize=10)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="png")
    finally:
        figure.clear()
    return path


def plot_color_curves(analysis, clouds: Sequence[ColorPointCloud], *, output: str | Path = "curves.png") -> Path:
    """Compare separate original-pixel clouds on the source and calibrated axes."""
    rgb, frame, _ = _analysis_parts(analysis)
    clouds = tuple(clouds)
    if not clouds or any(not isinstance(cloud, ColorPointCloud) for cloud in clouds):
        raise ValueError("Supply one or more ColorPointCloud results.")
    if any(cloud.image_shape != rgb.shape[:2] or cloud.frame.rectangle != frame.rectangle for cloud in clouds):
        raise ValueError("Clouds and analysis describe different image geometry.")
    path = Path(output)
    if path.suffix.lower() != ".png":
        raise ValueError("The curve plot output must have a .png extension.")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle as RectanglePatch

    figure = Figure(figsize=(12, 6), dpi=150, layout="constrained")
    FigureCanvasAgg(figure)
    try:
        source, calibrated = figure.subplots(1, 2)
        source.imshow(rgb, origin="upper", interpolation="nearest")
        palette = ("#e87500", "#00a6b2", "#a02bc4", "#3b8c28")
        for i, cloud in enumerate(clouds):
            color = palette[i % len(palette)]
            label = f"Curve {i + 1}: {len(cloud.x):,} pixels"
            source.scatter(cloud.pixels[:, 0], cloud.pixels[:, 1], s=.5, color=color, linewidths=0)
            calibrated.scatter(cloud.x, cloud.y, s=.7, color=color, linewidths=0, label=label)
        for region in dict.fromkeys(region for cloud in clouds for region in cloud.excluded_regions):
            source.add_patch(RectanglePatch((region.left, region.top), region.width, region.height,
                                            fill=False, edgecolor="#da9100", linestyle="--", linewidth=.8))
        source.set(title="Selected original pixels; dashed boxes = explicit exclusions")
        source.set_axis_off()
        calibration = clouds[0].calibration
        if calibration.x.scale == "log":
            calibrated.set_xscale("log")
        if calibration.y.scale == "log":
            calibrated.set_yscale("log")
        r = frame.rectangle
        calibrated.set_xlim(calibration.x.transform([r.left, r.right])[0])
        calibrated.set_ylim(calibration.y.transform([r.bottom, r.top])[0])
        calibrated.set(title="Separate calibrated pixel clouds", xlabel="x (axis units)", ylabel="y (axis units)")
        calibrated.grid(alpha=.2)
        calibrated.legend(fontsize=8)
        figure.suptitle("Visible pixels only; gaps are retained and CSV eps describes calibration uncertainty", fontsize=10)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="png")
    finally:
        figure.clear()
    return path
