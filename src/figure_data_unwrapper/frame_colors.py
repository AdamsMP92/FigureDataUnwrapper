"""Inspect original RGB along each frame side and conservatively refine bounds."""

from dataclasses import replace
import math

import numpy as np
from numpy.typing import ArrayLike

from .frame import FrameColorChange, FrameColorValidation, FrameDetection, FrameEdgeColor
from .preprocessing import _normalized_image
from .rectangle import Rectangle


_SIDES = ("left", "top", "right", "bottom")


def _rgb_image(image):
    source = np.asarray(image)
    if source.dtype == np.bool_ and source.ndim == 2:
        source = (~source).astype(float)
    elif np.issubdtype(source.dtype, np.integer):
        if not source.size or source.min() < 0 or source.max() > 255:
            raise ValueError("Integer image channels must be in [0, 255].")
        source = source.astype(float) / 255
    rgb = _normalized_image(source)
    return np.repeat(rgb[..., None], 3, axis=2) if rgb.ndim == 2 else rgb


def _runs(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []


def _sample_path(rgb, rectangle, side):
    horizontal = side in ("top", "bottom")
    start, end = (rectangle.left, rectangle.right) if horizontal else (rectangle.top, rectangle.bottom)
    # Corners join independently colored sides; do not compare their shared ink.
    margin = max(2, round(0.002 * (end - start)))
    if end - start <= 2 * margin:
        margin = 0
    tangent = np.arange(math.ceil(start + margin), math.floor(end - margin) + 1)
    normal = int(round(getattr(rectangle, side)))
    colors = rgb[normal, tangent] if horizontal else rgb[tangent, normal]
    return tangent, normal, colors


def _dominant_color(colors, tolerance):
    # Bins identify the dominant cluster only; all checks use original RGB.
    bins = np.rint(colors / (tolerance / 2)).astype(int)
    _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    dominant = np.median(colors[inverse == counts.argmax()], axis=0)
    nearby = np.max(abs(colors - dominant), axis=1) <= tolerance
    return np.median(colors[nearby], axis=0)


def _stroke_strength(rgb, rectangle, side, tangent, normal, colors, probe_radius):
    """A stroke differs from interpolation between both transverse flanks.

    Smooth shading and step-edge antialiasing lie on the RGB segment between
    their flank colors. A narrow dark, bright, or colored stroke departs from it.
    """
    horizontal = side in ("top", "bottom")
    extent = rgb.shape[0] if horizontal else rgb.shape[1]
    strength = np.zeros(len(tangent))
    radii = sorted({1, 2, 4, probe_radius})
    for radius in radii:
        if radius > probe_radius or normal - radius < 0 or normal + radius >= extent:
            continue
        first = rgb[normal - radius, tangent] if horizontal else rgb[tangent, normal - radius]
        last = rgb[normal + radius, tangent] if horizontal else rgb[tangent, normal + radius]
        direction = last - first
        denominator = np.sum(direction * direction, axis=1)
        fraction = np.divide(np.sum((colors - first) * direction, axis=1), denominator,
                             out=np.zeros_like(denominator), where=denominator > 1e-12)
        interpolated = first + np.clip(fraction, 0, 1)[:, None] * direction
        strength = np.maximum(strength, np.max(abs(colors - interpolated), axis=1))
    return strength


def _edge_check(rgb, rectangle, side, color_tolerance, min_contrast, interruption, probe_radius):
    tangent, normal, colors = _sample_path(rgb, rectangle, side)
    if len(tangent) == 0:
        return FrameEdgeColor(side, getattr(rectangle, side), (0., 0., 0.), 0, 0, 0, "unobserved")
    dominant = _dominant_color(colors, color_tolerance)
    same = np.max(abs(colors - dominant), axis=1) <= color_tolerance
    fraction = float(same.mean())
    limit = max(3, math.ceil(0.01 * len(tangent))) if interruption is None else interruption
    changes = []
    for run in _runs(np.flatnonzero(~same)):
        at_end = run[0] == 0 or run[-1] == len(tangent) - 1
        if len(run) <= limit:
            allowed, reason = True, "short_interruption"
        elif not at_end and len(run) <= 0.20 * len(tangent) and fraction >= 0.75:
            allowed, reason = True, "bounded_occlusion"
        else:
            allowed, reason = False, "endpoint_color_change" if at_end else "persistent_color_change"
        changes.append(FrameColorChange(float(tangent[run[0]]), float(tangent[run[-1]]),
                                         tuple(float(v) for v in np.median(colors[run], axis=0)),
                                         allowed, reason))
    strength = _stroke_strength(rgb, rectangle, side, tangent, normal, colors, probe_radius)
    support = float(np.mean(strength >= min_contrast))
    if support >= 0.5 and any(not change.allowed for change in changes):
        status = "color_change"
    elif support < 0.75:
        status = "unobserved"
    else:
        status = "interrupted" if changes else "valid"
    jumps = tangent[1:][np.max(abs(np.diff(colors, axis=0)), axis=1) > color_tolerance]
    return FrameEdgeColor(
        side, getattr(rectangle, side), tuple(float(v) for v in dominant),
        fraction, support, float(np.median(strength)), status, tuple(changes),
        tuple(float(value) for value in jumps),
    )


def _inspect(rgb, rectangle, options):
    return tuple(_edge_check(rgb, rectangle, side, *options) for side in _SIDES)


def _score(edges):
    return (
        -sum(edge.status == "color_change" for edge in edges),
        sum(edge.plausible for edge in edges),
        sum(edge.stroke_strength for edge in edges if edge.plausible),
    )


def _correction_signals(edges, rectangle):
    """Endpoint color changes point to an orthogonal boundary, not a new color."""
    signals = {side: [] for side in _SIDES}
    for edge in edges:
        if edge.stroke_fraction < 0.5 or edge.consistent_fraction < 0.6:
            continue
        horizontal = edge.side in ("top", "bottom")
        start, end = (rectangle.left, rectangle.right) if horizontal else (rectangle.top, rectangle.bottom)
        margin = max(2, round(0.002 * (end - start)))
        first_side, last_side = ("left", "right") if horizontal else ("top", "bottom")
        for change in edge.changes:
            if change.allowed or change.reason != "endpoint_color_change":
                continue
            if change.start <= math.ceil(start + margin):
                signals[first_side].append(change.end + 1)
            if change.end >= math.floor(end - margin):
                signals[last_side].append(change.start - 1)
    return signals


def check_frame_colors(
    image: ArrayLike, frame: FrameDetection | None, *, correct: bool = True,
    color_tolerance: float = 0.1, min_line_contrast: float = 0.03,
    max_interruption_px: int | None = None, probe_radius: int = 8,
    search_fraction: float = 0.1,
) -> FrameDetection | None:
    """Check each side in original RGB and refine transition-supported bounds.

    Different sides may have entirely different colors. Each side gets its own
    dominant RGB reference. Corner joins, short deviations, and bounded A-B-A
    occlusions are allowed. Sustained endpoint changes are suspect and nominate
    an orthogonal boundary for correction. Constant color alone is insufficient:
    transverse contrast must support a narrow stroke, including white strokes.

    Corrections require a full-length, color-consistent stroke near the nominated
    transition and improve the complete rectangle without damaging good sides.
    Missing/invisible sides are kept inferred; no white-on-white line is invented.
    Inspect frame.color_check for before/after profiles, allowed changes, jumps,
    and unresolved cases. Bounds move by at most search_fraction of the initial
    perpendicular span; at most four improving updates are made.
    """
    if not 0 < color_tolerance < 1 or not 0 < min_line_contrast < 1 or not 0 < search_fraction <= 0.5:
        raise ValueError("Invalid color/contrast tolerance or search_fraction.")
    if not isinstance(correct, (bool, np.bool_)):
        raise ValueError("correct must be a boolean.")
    for name, value in (("probe_radius", probe_radius), ("max_interruption_px", max_interruption_px)):
        if value is None and name == "max_interruption_px":
            continue
        minimum = 0 if name == "max_interruption_px" else 1
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if frame is None:
        return None
    if not isinstance(frame, FrameDetection):
        raise TypeError("frame must be a FrameDetection or None.")
    rgb = _rgb_image(image)
    original = frame.rectangle
    height, width = rgb.shape[:2]
    if not (0 <= original.left < original.right < width and 0 <= original.top < original.bottom < height):
        raise ValueError("Frame bounds lie outside the image.")
    options = (color_tolerance, min_line_contrast, max_interruption_px, probe_radius)
    original_edges = _inspect(rgb, original, options)
    rectangle, edges = original, original_edges
    if correct:
        for _ in range(4):
            signals = _correction_signals(edges, rectangle)
            best_rectangle, best_edges = rectangle, edges
            for side, coordinates in signals.items():
                if not coordinates or np.ptp(coordinates) > 2 * probe_radius:
                    continue
                target = float(np.median(coordinates))
                original_coordinate = getattr(original, side)
                span = original.height if side in ("top", "bottom") else original.width
                radius = max(4, search_fraction * span)
                # An out-of-range transition cannot justify a partial move to
                # shading or an unrelated ridge inside the allowed window.
                if abs(target - original_coordinate) > radius:
                    continue
                candidates = []
                for position in range(math.floor(target - 2 * probe_radius), math.ceil(target + 2 * probe_radius) + 1):
                    if abs(position - original_coordinate) > radius:
                        continue
                    bounds = {name: getattr(rectangle, name) for name in _SIDES}
                    bounds[side] = float(position)
                    if not (0 <= bounds["left"] < bounds["right"] < width and 0 <= bounds["top"] < bounds["bottom"] < height):
                        continue
                    proposal = Rectangle(**bounds)
                    edge = _edge_check(rgb, proposal, side, *options)
                    if edge.plausible:
                        candidates.append((edge.stroke_strength, proposal))
                # Recheck the whole rectangle only for the strongest local ridges.
                for _, proposal in sorted(candidates, key=lambda item: item[0], reverse=True)[:3]:
                    proposed_edges = _inspect(rgb, proposal, options)
                    if any(old.plausible and not new.plausible for old, new in zip(edges, proposed_edges)):
                        continue
                    if _score(proposed_edges) > _score(best_edges):
                        best_rectangle, best_edges = proposal, proposed_edges
            if best_rectangle == rectangle:
                break
            rectangle, edges = best_rectangle, best_edges
    visible = tuple(edge.side for edge in edges if edge.plausible)
    status = "suspect" if any(edge.status == "color_change" for edge in edges) else (
        "valid" if len(visible) == 4 else "partial" if len(visible) >= 2 else "rejected"
    )
    corrected = rectangle != original
    notes = []
    if corrected:
        changes = ", ".join(f"{side} {getattr(original, side):g} -> {getattr(rectangle, side):g}"
                            for side in _SIDES if getattr(original, side) != getattr(rectangle, side))
        notes.append(f"RGB frame correction: {changes} px.")
    if status in ("suspect", "rejected"):
        notes.append("RGB frame check unresolved; tick/grid extraction is withheld.")
    elif status == "partial":
        notes.append("RGB frame check: some sides remain inferred or invisible.")
    validation = FrameColorValidation(original, original_edges, edges, corrected, status, tuple(notes))
    return replace(frame, rectangle=rectangle, visible_sides=visible, color_check=validation)
