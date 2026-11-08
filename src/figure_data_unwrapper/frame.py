"""Infer axis-aligned plot rectangles from finite line segments."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .preprocessing import _normalized_image, foreground_mask
from .rectangle import Rectangle


_SIDES = ("left", "top", "right", "bottom")


@dataclass(frozen=True)
class FrameColorChange:
    """A different-color interval along one side, in tangential pixel coordinates."""

    start: float
    end: float
    color: tuple[float, float, float]
    allowed: bool
    reason: str


@dataclass(frozen=True)
class FrameEdgeColor:
    """Original-RGB evidence for one side; colors are never shared across sides."""

    side: str
    coordinate: float
    color: tuple[float, float, float]
    consistent_fraction: float
    stroke_fraction: float
    stroke_strength: float
    status: str
    changes: tuple[FrameColorChange, ...] = ()
    jump_positions: tuple[float, ...] = ()

    @property
    def plausible(self) -> bool:
        return self.status in ("valid", "interrupted")


@dataclass(frozen=True)
class FrameColorValidation:
    """Independent side checks before/after refinement, with normalized RGB."""

    original_rectangle: Rectangle
    original_edges: tuple[FrameEdgeColor, ...]
    edges: tuple[FrameEdgeColor, ...]
    corrected: bool
    status: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameDetection:
    """Pixel geometry and the sides backed by full-length line evidence."""

    rectangle: Rectangle
    visible_sides: tuple[str, ...]
    color_check: FrameColorValidation | None = None

    @property
    def inferred_sides(self) -> tuple[str, ...]:
        return tuple(side for side in _SIDES if side not in self.visible_sides)


@dataclass(frozen=True)
class _Segment:
    horizontal: bool
    coordinate: float
    start: float
    end: float
    thickness: int


def _line_candidates(
    mask: NDArray[np.bool_], minimum_length: float, min_coverage: float,
    max_candidates: int, *, horizontal: bool,
) -> list[_Segment]:
    support = mask.sum(axis=1)
    rows = np.flatnonzero(support >= minimum_length * min_coverage)
    if rows.size == 0:
        return []
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    segments = []
    max_gap = max(3, round(0.005 * mask.shape[1]))
    for group in groups:
        # Trace the center of the stroke: an any-pixel union across its width
        # can bridge a corner tick to nearby text and extend the spine into a
        # tick label. Small genuine internal gaps are still permitted below.
        profile = mask[group[len(group) // 2]]
        points = np.flatnonzero(profile)
        runs = np.split(points, np.flatnonzero(np.diff(points) > max_gap + 1) + 1)
        valid = [run for run in runs if (
            run[-1] - run[0] >= minimum_length
            and len(run) / (run[-1] - run[0] + 1) >= min_coverage
            and run[-1] - run[0] >= 5 * len(group)
        )]
        if not valid:
            continue
        run = max(valid, key=lambda item: item[-1] - item[0])
        segments.append(_Segment(
            horizontal, float((group[0] + group[-1]) / 2),
            float(run[0]), float(run[-1]), len(group),
        ))
    return sorted(segments, key=lambda line: line.end - line.start, reverse=True)[
        :max_candidates
    ]


def _tolerance(first: _Segment, second: _Segment) -> float:
    return (first.thickness + second.thickness) / 2 + 1


def _endpoint_slack(line: _Segment) -> float:
    """Permit short corner ticks continuing an otherwise full-length spine."""
    stroke_limit = max(12, 4 * line.thickness)
    return line.thickness / 2 + max(2, min(stroke_limit, 0.08 * (line.end - line.start)))


def _propose(first: _Segment, second: _Segment) -> Rectangle | None:
    tolerance = _tolerance(first, second)
    if first.horizontal == second.horizontal:
        if (
            abs(first.start - second.start) > tolerance
            or abs(first.end - second.end) > tolerance
            or abs(first.coordinate - second.coordinate) <= tolerance
        ):
            return None
        start = (first.start + second.start) / 2
        end = (first.end + second.end) / 2
        low, high = sorted((first.coordinate, second.coordinate))
        return Rectangle(start, low, end, high) if first.horizontal else Rectangle(
            low, start, high, end
        )

    h, v = (first, second) if first.horizontal else (second, first)
    if abs(v.coordinate - h.start) <= _endpoint_slack(h):
        left, right = v.coordinate, h.end
    elif abs(v.coordinate - h.end) <= _endpoint_slack(h):
        left, right = h.start, v.coordinate
    else:
        return None
    if abs(h.coordinate - v.start) <= _endpoint_slack(v):
        top, bottom = h.coordinate, v.end
    elif abs(h.coordinate - v.end) <= _endpoint_slack(v):
        top, bottom = v.start, h.coordinate
    else:
        return None
    if left >= right or top >= bottom:
        return None
    return Rectangle(left, top, right, bottom)


def _matching_sides(
    rectangle: Rectangle, segments: list[_Segment]
) -> dict[str, _Segment]:
    matches = {}
    for side in _SIDES:
        horizontal = side in ("top", "bottom")
        start, end = (
            (rectangle.left, rectangle.right) if horizontal
            else (rectangle.top, rectangle.bottom)
        )
        coordinate = getattr(rectangle, side)
        candidates = [line for line in segments if (
            line.horizontal == horizontal
            and abs(line.coordinate - coordinate) <= line.thickness / 2 + 2
            and start - _endpoint_slack(line) <= line.start <= start + line.thickness / 2 + 2
            and end - line.thickness / 2 - 2 <= line.end <= end + _endpoint_slack(line)
            # At least one end must be anchored. Four long grid lines crossing
            # a smaller interior box must not masquerade as four frame sides.
            and min(abs(line.start - start), abs(line.end - end)) <= line.thickness / 2 + 2
        )]
        if candidates:
            matches[side] = min(candidates, key=lambda line: abs(line.coordinate - coordinate))
    return matches


def fit_frame(
    foreground: ArrayLike, *, min_side_fraction: float = 0.25,
    min_coverage: float = 0.9, max_candidates: int = 20,
) -> FrameDetection | None:
    """Fit a rectangle to a boolean foreground mask; True means line evidence.

    Adjacent segments must meet at their endpoints; opposite segments must
    agree on their longitudinal extents. At least two full sides are required.
    Missing sides are inferred from the finite segment endpoints. Among valid
    proposals the greatest number of supported sides wins, then largest area.
    Short line extensions at corners may be ticks; intersecting side positions
    take precedence over these extensions when establishing the bounds.

    This assumes a single axis-aligned plot with full-length spines and no
    larger competing rectangular objects. Cropped or shortened spines do not
    uniquely establish the original plot bounds. No data-coordinate calibration
    is performed. Search is bounded to max_candidates segments per orientation.
    """
    mask = np.asarray(foreground)
    if mask.ndim != 2 or mask.size == 0 or mask.dtype != np.bool_:
        raise ValueError("Expected a nonempty 2D boolean foreground mask.")
    if not 0 < min_side_fraction <= 1 or not 0 < min_coverage <= 1:
        raise ValueError("min_side_fraction and min_coverage must be in (0, 1].")
    if not isinstance(max_candidates, int) or max_candidates < 2:
        raise ValueError("max_candidates must be an integer of at least 2.")
    height, width = mask.shape
    segments = _line_candidates(
        mask, min_side_fraction * width, min_coverage, max_candidates, horizontal=True
    ) + _line_candidates(
        mask.T, min_side_fraction * height, min_coverage, max_candidates, horizontal=False
    )
    best = None
    best_score = (0, 0.0)
    for first, second in combinations(segments, 2):
        rectangle = _propose(first, second)
        if rectangle is None:
            continue
        # Prefer measured centerlines wherever a side is actually present.
        matches = _matching_sides(rectangle, segments)
        bounds = {
            side: matches[side].coordinate if side in matches else getattr(rectangle, side)
            for side in _SIDES
        }
        if bounds["left"] >= bounds["right"] or bounds["top"] >= bounds["bottom"]:
            continue
        rectangle = Rectangle(**bounds)
        matches = _matching_sides(rectangle, segments)
        if (
            len(matches) < 2
            or rectangle.width < min_side_fraction * width
            or rectangle.height < min_side_fraction * height
        ):
            continue
        score = (len(matches), rectangle.width * rectangle.height)
        if score > best_score:
            best = FrameDetection(rectangle, tuple(matches))
            best_score = score
    return best


def detect_frame(
    image: ArrayLike, *, darkness_threshold: float | None = None,
    min_contrast: float = 0.03, min_side_fraction: float = 0.25,
    min_coverage: float = 0.9, max_candidates: int = 20,
) -> Rectangle | None:
    """Infer a rectangle, then validate and refine its sides in original RGB.

    By default use foreground_mask, supporting grayscale, RGB, RGBA and boolean
    masks. To reproduce the previous dark-pixel preprocessing, explicitly pass
    darkness_threshold (e.g. 0.5). Color input is then reduced to luminance.
    Return None for unresolved RGB inconsistencies. Use check_frame_colors on
    fit_frame(foreground_mask(image)) for side metadata and color diagnostics;
    fit_frame alone performs only the binary geometric stage.
    """
    if darkness_threshold is None:
        mask = foreground_mask(image, min_contrast=min_contrast)
    else:
        if not 0 < darkness_threshold < 1:
            raise ValueError("darkness_threshold must be between 0 and 1.")
        values = _normalized_image(image)
        if values.ndim == 3:
            values = values @ np.array([0.2126, 0.7152, 0.0722])
        mask = values < darkness_threshold
    result = fit_frame(
        mask, min_side_fraction=min_side_fraction, min_coverage=min_coverage,
        max_candidates=max_candidates,
    )
    from .frame_colors import check_frame_colors

    result = check_frame_colors(image, result, min_line_contrast=min_contrast)
    if result is None or result.color_check.status in ("suspect", "rejected"):
        return None
    return result.rectangle
