"""Find perpendicular tick branches and grid lines relative to a fitted frame."""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike

from .frame import FrameDetection, fit_frame
from .frame_colors import check_frame_colors
from .preprocessing import _foreground_contrast


@dataclass(frozen=True)
class TickMark:
    """One short branch at a frame side, with lengths measured in pixels."""

    side: str
    position: float
    inward_length: float
    outward_length: float


@dataclass(frozen=True)
class AxisTick:
    """A pixel position, merging corresponding marks on opposing sides."""

    position: float
    marks: tuple[TickMark, ...]
    value: float | None = None

    @property
    def sides(self) -> tuple[str, ...]:
        return tuple(mark.side for mark in self.marks)


@dataclass(frozen=True)
class TickOutlier:
    """Separated evidence; reason need not imply that the image mark is false."""

    tick: AxisTick
    reason: str
    residual_pixels: float | None = None


@dataclass(frozen=True)
class AxisTickValidation:
    """Axis-value or spacing check; pixel_slope acts on value or log(value)."""

    scale: str
    status: str
    reference_values: tuple[float, ...] | None
    tolerance_pixels: float
    missing_values: tuple[float, ...] = ()
    pixel_slope: float | None = None
    pixel_offset: float | None = None
    rms_error_pixels: float | None = None
    spacing_pixels: float | None = None


@dataclass(frozen=True)
class GridLine:
    """A grid-like line; coverage and contrast are measurements, not confidence."""

    orientation: str
    position: float
    start: float
    end: float
    coverage: float
    contrast: float
    width: int
    frame_sides: tuple[str, ...] = ()


@dataclass(frozen=True)
class TickGridDetection:
    frame: FrameDetection
    x_ticks: tuple[AxisTick, ...]
    y_ticks: tuple[AxisTick, ...]
    vertical_grid: tuple[GridLine, ...]
    horizontal_grid: tuple[GridLine, ...]
    expected_x_ticks: int | None
    expected_y_ticks: int | None
    notes: tuple[str, ...]
    rejected_x_ticks: tuple[TickOutlier, ...] = ()
    rejected_y_ticks: tuple[TickOutlier, ...] = ()
    x_validation: AxisTickValidation | None = None
    y_validation: AxisTickValidation | None = None

    @property
    def x_count_matches(self) -> bool | None:
        return None if self.expected_x_ticks is None else len(self.x_ticks) == self.expected_x_ticks

    @property
    def y_count_matches(self) -> bool | None:
        return None if self.expected_y_ticks is None else len(self.y_ticks) == self.expected_y_ticks


def _groups(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []


def _edge_band(mask, rectangle, side):
    """Locate the shaft so its stroke width is excluded from branch scans."""
    horizontal = side in ("top", "bottom")
    oriented = mask if horizontal else mask.T
    start, end = (rectangle.left, rectangle.right) if horizontal else (rectangle.top, rectangle.bottom)
    coordinate = getattr(rectangle, side)
    along = np.arange(max(0, math.ceil(start)), min(oriented.shape[1], math.floor(end) + 1))
    normal = np.arange(max(0, math.floor(coordinate) - 6),
                       min(oriented.shape[0], math.ceil(coordinate) + 7))
    support = oriented[np.ix_(normal, along)].mean(axis=1)
    bands = _groups(normal[support >= 0.65])
    bands = [band for band in bands if np.min(np.abs(band - coordinate)) <= 1.5]
    if bands:
        band = min(bands, key=lambda item: np.min(np.abs(item - coordinate)))
        return int(band[0]), int(band[-1])
    return math.ceil(coordinate), math.floor(coordinate)


def _grid_lines(contrast, mask, inner, *, horizontal, min_coverage, max_gap):
    """Seed at opposite frame boundaries, then follow perpendicular branches.

    The region starts immediately beyond each frame stroke. Only positions with
    foreground near both boundaries are candidates. Trace each candidate at its
    fixed perpendicular coordinate, rejecting any gap longer than the permitted
    dash gap. An interior row/column's total foreground fraction cannot seed a
    line: detached text and legend fragments never enter the search this way.
    """
    left, top, right, bottom = inner
    region = mask[top:bottom + 1, left:right + 1]
    values = contrast[top:bottom + 1, left:right + 1]
    if horizontal:
        region, values = region.T, values.T
    # Rows follow the inward normal; columns walk along the frame side.
    length = region.shape[0]
    if length < 3 or region.shape[1] == 0:
        return ()
    gap_limit = max(4, math.ceil(0.01 * length)) if max_gap is None else int(max_gap)
    root_depth = min(length, gap_limit + 1)
    from_first_side = region[:root_depth].any(axis=0)
    from_last_side = region[-root_depth:].any(axis=0)
    branches = np.flatnonzero(from_first_side & from_last_side)
    traced = []
    for column in branches:
        points = np.flatnonzero(region[:, column])
        # Boundary support alone is insufficient: short tick stubs must not
        # be joined through a distant legend or data segment in the interior.
        if len(points) / length < min_coverage:
            continue
        if np.any(np.diff(points) - 1 > gap_limit):
            continue
        traced.append(column)
    lines = []
    for group in _groups(np.asarray(traced, dtype=int)):
        profile = region[:, group].any(axis=1)
        points = np.flatnonzero(profile)
        foreground_values = values[:, group][region[:, group]]
        position_offset = top if horizontal else left
        longitudinal_offset = left if horizontal else top
        lines.append(GridLine(
            "horizontal" if horizontal else "vertical",
            float(position_offset + (group[0] + group[-1]) / 2),
            float(longitudinal_offset + points[0]),
            float(longitudinal_offset + points[-1]),
            float(profile.mean()), float(np.median(foreground_values)), len(group),
            ("left", "right") if horizontal else ("top", "bottom"),
        ))
    return tuple(lines)


def _short_branch_lengths(strip, normal, coordinate, minimum, limit):
    """Require attachment to the shaft and stop at the first real gap."""
    count = strip.shape[1]
    if len(normal) == 0:
        return np.zeros(count)
    roots = strip[:2].any(axis=0)
    starts = strip.argmax(axis=0)
    after_start = np.arange(len(normal))[:, None] >= starts
    gaps = ~strip & after_start
    stops = np.where(gaps.any(axis=0), gaps.argmax(axis=0), len(normal))
    runs = stops - starts
    valid = roots & (runs >= minimum) & (runs < limit)
    ends = normal[np.clip(stops - 1, 0, len(normal) - 1)]
    return np.where(valid, np.abs(ends - coordinate) + 0.5, 0.0)


def _side_marks(contrast, rectangle, side, band, grid, min_contrast, min_tick_length):
    horizontal_side = side in ("top", "bottom")
    values = contrast if horizontal_side else contrast.T
    start, end = (rectangle.left, rectangle.right) if horizontal_side else (rectangle.top, rectangle.bottom)
    span = rectangle.height if horizontal_side else rectangle.width
    coordinate = getattr(rectangle, side)
    tangent = np.arange(max(0, math.floor(start) - 1),
                        min(values.shape[1], math.ceil(end) + 2))
    limit = max(8, math.ceil(0.10 * span))
    inward_direction = 1 if side in ("top", "left") else -1

    def sample(direction):
        first = band[1] + 1 if direction > 0 else band[0] - 1
        normal = first + direction * np.arange(limit + 1)
        normal = normal[(normal >= 0) & (normal < values.shape[0])]
        return values[np.ix_(normal, tangent)], normal

    inside, inside_normal = sample(inward_direction)
    outside, outside_normal = sample(-inward_direction)
    thresholds = np.full(len(tangent), min_contrast)
    for line in grid:
        on_line = np.abs(tangent - line.position) <= line.width / 2 + 1
        # At grid positions, only a stronger short stroke is separate evidence
        # for an inward tick. Equal-color overlaid strokes are inseparable.
        thresholds[on_line] = max(min_contrast, line.contrast * 1.5, line.contrast + 0.03)
    inward = _short_branch_lengths(
        inside >= thresholds, inside_normal, coordinate, min_tick_length, limit
    )
    outward = _short_branch_lengths(
        outside >= min_contrast, outside_normal, coordinate, min_tick_length, limit
    )
    marks = []
    for group in _groups(np.flatnonzero((inward > 0) | (outward > 0))):
        inner_length = float(inward[group].max())
        outer_length = float(outward[group].max())
        if max(inner_length, outer_length) < 1.2 * len(group):
            continue  # Reject broad blobs attached to the frame.
        marks.append(TickMark(
            side, float((tangent[group[0]] + tangent[group[-1]]) / 2),
            inner_length, outer_length,
        ))
    return marks


def _merge_marks(marks):
    clusters = []
    for mark in sorted(marks, key=lambda item: item.position):
        if (
            clusters and abs(mark.position - np.mean([m.position for m in clusters[-1]])) <= 1.5
            and mark.side not in {m.side for m in clusters[-1]}
        ):
            clusters[-1].append(mark)
        else:
            clusters.append([mark])
    return tuple(AxisTick(float(np.mean([m.position for m in cluster])), tuple(cluster))
                 for cluster in clusters)


def _apply_count_hint(ticks, expected, axis, notes):
    if expected is None or len(ticks) == expected:
        return ticks
    if expected > 0 and len(ticks) > expected:
        strength = lambda tick: max(max(mark.inward_length, mark.outward_length) for mark in tick.marks)
        ranked = sorted(ticks, key=strength, reverse=True)
        if strength(ranked[expected - 1]) - strength(ranked[expected]) >= 0.75:
            notes.append(f"{axis}: selected {expected} longer ticks from {len(ticks)} observed positions.")
            return tuple(sorted(ranked[:expected], key=lambda tick: tick.position))
        notes.append(f"{axis}: requested {expected} ticks, found {len(ticks)} with ambiguous lengths; kept all.")
    else:
        notes.append(f"{axis}: requested {expected} ticks, observed {len(ticks)}; no positions invented or discarded.")
    return ticks


def detect_ticks_and_grid(
    image: ArrayLike, *, frame: FrameDetection | None = None,
    expected_x_ticks: int | None = None, expected_y_ticks: int | None = None,
    min_contrast: float = 0.03, min_tick_length: float = 2,
    grid_min_coverage: float = 0.3, grid_max_gap: int | None = None,
) -> TickGridDetection | None:
    """Automatically find ticks and grid-like lines after frame reconstruction.

    Accept a successful fit_frame result or reconstruct the frame automatically.
    Validate/refine unchecked frames in original RGB before extracting branches.
    Return None when reconstruction fails or color checks remain unresolved.
    Pixel positions are sorted left to right for x, top to bottom for y.
    No spacing or axis-scale model is assumed.

    Short perpendicular branches identify ticks. Grid search starts with branch
    support immediately inside opposite frame sides, then traces straight across
    the plot, allowing only short dash gaps. grid_max_gap sets that allowance in
    pixels; None uses max(4, ceil(1% of the interior span)). It applies to both
    boundary attachment and internal gaps. Detached interior strokes cannot seed
    a grid line. GridLine.frame_sides records the supporting frame boundaries.
    Opposing tick marks are merged, and frame shafts are excluded.
    Count hints select clearly longer ticks when possible, otherwise retain the
    evidence and describe the mismatch in notes. Hints never synthesize positions.
    Tick labels/values and major/minor semantics are not recognized here.
    """
    for name, value in (("expected_x_ticks", expected_x_ticks), ("expected_y_ticks", expected_y_ticks)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer or None.")
    if not 0 < min_contrast < 1 or not 0 < grid_min_coverage <= 1:
        raise ValueError("Invalid contrast or grid coverage threshold.")
    if not math.isfinite(min_tick_length) or min_tick_length <= 0:
        raise ValueError("min_tick_length must be finite and positive.")
    if grid_max_gap is not None and (
        isinstance(grid_max_gap, (bool, np.bool_))
        or not isinstance(grid_max_gap, (int, np.integer)) or grid_max_gap < 0
    ):
        raise ValueError("grid_max_gap must be a nonnegative integer or None.")
    contrast = _foreground_contrast(image)
    mask = contrast >= min_contrast
    if frame is None:
        frame = fit_frame(mask)
        if frame is None:
            return None
    if not isinstance(frame, FrameDetection):
        raise TypeError("frame must be a successful fit_frame result.")
    if frame.color_check is not None and frame.color_check.status in ("suspect", "rejected"):
        return None
    if len(set(frame.visible_sides)) < 2 or not set(frame.visible_sides) <= {"top", "bottom", "left", "right"}:
        raise ValueError("The supplied frame must have at least two observed sides.")
    r = frame.rectangle
    height, width = mask.shape
    if not (0 <= r.left < r.right < width and 0 <= r.top < r.bottom < height):
        raise ValueError("The supplied frame lies outside this image.")
    if frame.color_check is None:
        frame = check_frame_colors(image, frame, min_line_contrast=min_contrast)
    if frame.color_check.status in ("suspect", "rejected"):
        return None
    r = frame.rectangle
    bands = {side: _edge_band(mask, r, side) for side in ("top", "bottom", "left", "right")}
    inner = (
        max(math.floor(r.left) + 1, bands["left"][1] + 1),
        max(math.floor(r.top) + 1, bands["top"][1] + 1),
        min(math.ceil(r.right) - 1, bands["right"][0] - 1),
        min(math.ceil(r.bottom) - 1, bands["bottom"][0] - 1),
    )
    if inner[0] > inner[2] or inner[1] > inner[3]:
        raise ValueError("The frame has no resolvable interior.")
    vertical = _grid_lines(contrast, mask, inner, horizontal=False,
                           min_coverage=grid_min_coverage, max_gap=grid_max_gap)
    horizontal = _grid_lines(contrast, mask, inner, horizontal=True,
                             min_coverage=grid_min_coverage, max_gap=grid_max_gap)
    x_marks, y_marks = [], []
    for side in ("bottom", "top", "left", "right"):
        marks = _side_marks(
            contrast, r, side, bands[side],
            vertical if side in ("bottom", "top") else horizontal,
            min_contrast, min_tick_length,
        )
        (x_marks if side in ("bottom", "top") else y_marks).extend(marks)
    notes = []
    x_ticks = _apply_count_hint(_merge_marks(x_marks), expected_x_ticks, "x", notes)
    y_ticks = _apply_count_hint(_merge_marks(y_marks), expected_y_ticks, "y", notes)
    return TickGridDetection(frame, x_ticks, y_ticks, vertical, horizontal,
                             expected_x_ticks, expected_y_ticks, tuple(notes))
