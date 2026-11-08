"""Reference-marker matching with color masking and overlapping-template fits."""

from dataclasses import dataclass
from pathlib import Path
import math

from matplotlib.image import imread
import numpy as np

from .extraction import CoordinateCalibration, _analysis_parts, _color_rgb, _positive_number, calibrate_axes
from .frame import FrameDetection
from .preprocessing import _normalized_image
from .rectangle import Rectangle


@dataclass(frozen=True)
class MarkerMatch:
    pixel_x: float
    pixel_y: float
    scale: float
    score: float
    color_support: float
    visible_fraction: float
    mse: float
    unique_gain: float
    overlaps_neighbor: bool


@dataclass(frozen=True)
class MarkerPointCloud:
    """One point per marker hypothesis; eps contains calibration uncertainty only.

    Unpack x, y, eps, with eps shaped (N, 2). Match scores are diagnostics, not
    probabilities. Center-localization uncertainty is not estimated by eps.
    """

    x: np.ndarray
    y: np.ndarray
    eps: np.ndarray
    matches: tuple[MarkerMatch, ...]
    color: tuple[float, float, float]
    frame: FrameDetection
    calibration: CoordinateCalibration
    image_shape: tuple[int, int]
    template_shape: tuple[int, int]
    candidate_count: int
    excluded_regions: tuple[Rectangle, ...]
    notes: tuple[str, ...]

    @property
    def pixels(self):
        return np.array([(m.pixel_x, m.pixel_y) for m in self.matches], dtype=float).reshape(-1, 2)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.eps

    def to_array(self):
        """x,y,eps_x,eps_y,pixel_x,pixel_y,scale,score,support,visible,mse,gain,overlap."""
        quality = np.array([(m.scale, m.score, m.color_support, m.visible_fraction,
                             m.mse, m.unique_gain, m.overlaps_neighbor) for m in self.matches], dtype=float).reshape(-1, 7)
        return np.column_stack((self.x, self.y, self.eps, self.pixels, quality))


@dataclass(frozen=True)
class _Template:
    ink: np.ndarray
    anchor_x: float
    anchor_y: float
    scale: float
    energy: float
    diameter: float


@dataclass(frozen=True)
class _Placement:
    top: int
    left: int
    template: int
    score: float


def _rgb(source):
    source = np.asarray(source)
    if np.issubdtype(source.dtype, np.integer):
        if not source.size or source.min() < 0 or source.max() > 255:
            raise ValueError("Integer template channels must be in [0, 255].")
        source = source.astype(float) / 255
    rgb = _normalized_image(source)
    return np.repeat(rgb[..., None], 3, axis=2) if rgb.ndim == 2 else rgb


def _project(rgb, background, color, tolerance):
    direction = color - background
    ink = np.clip(np.sum((rgb - background) * direction, axis=2) / np.dot(direction, direction), 0, 1)
    residual = np.linalg.norm(rgb - (background + ink[..., None] * direction), axis=2)
    # Foreign-colored overdraw is unknown, rather than evidence of background.
    return ink, (residual <= tolerance).astype(float)


def _prepare(source, color, tolerance, scales, anchor):
    rgb = _rgb(imread(source) if isinstance(source, (str, Path)) else source)
    if min(rgb.shape[:2]) < 3:
        raise ValueError("The template must be at least 3 by 3 pixels.")
    perimeter = np.concatenate((rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]))
    background = np.median(perimeter, axis=0)
    contrast = np.linalg.norm(rgb - background, axis=2)
    if contrast.max() < 0.05:
        raise ValueError("The reference marker has insufficient background contrast.")
    target = np.median(rgb[contrast >= .8 * contrast.max()], axis=0) if color is None else _color_rgb(color)
    if np.linalg.norm(target - background) < 0.05:
        raise ValueError("Marker and background colors are indistinguishable.")
    ink, weight = _project(rgb, background, target, tolerance)
    ink *= weight
    yy, xx = np.nonzero(ink > .1)
    if len(xx) < 3:
        raise ValueError("No usable marker foreground in the template.")
    if anchor is None:
        y, x = np.indices(ink.shape)
        ax, ay = float(np.sum(x * ink) / ink.sum()), float(np.sum(y * ink) / ink.sum())
    else:
        position = np.asarray(anchor, dtype=float)
        if (position.shape != (2,) or not np.isfinite(position).all()
                or not 0 <= position[0] < rgb.shape[1] or not 0 <= position[1] < rgb.shape[0]):
            raise ValueError("anchor must be an (x, y) position inside the original template.")
        ax, ay = position
    top, left = max(0, yy.min() - 4), max(0, xx.min() - 4)
    cropped = ink[top:yy.max() + 5, left:xx.max() + 5]
    templates = []
    for scale in scales:
        h, w = (max(3, round(n * scale)) for n in cropped.shape)
        resized = np.array([np.interp(np.linspace(0, cropped.shape[1] - 1, w),
                                     np.arange(cropped.shape[1]), row) for row in cropped])
        resized = np.array([np.interp(np.linspace(0, cropped.shape[0] - 1, h),
                                     np.arange(cropped.shape[0]), row) for row in resized.T]).T
        xscale, yscale = (w - 1) / (cropped.shape[1] - 1), (h - 1) / (cropped.shape[0] - 1)
        templates.append(_Template(resized, (ax - left) * xscale, (ay - top) * yscale,
                                   scale, float(np.sum(resized ** 2)),
                                   max(np.ptp(xx) + 1, np.ptp(yy) + 1) * scale))
    return templates, background, target, rgb.shape[:2]


def _correlate(image, kernel):
    """Valid correlation; FFTs avoid materializing all sliding image windows."""
    h, w = image.shape
    kh, kw = kernel.shape
    shape = tuple(1 << (n - 1).bit_length() for n in (h + kh - 1, w + kw - 1))
    result = np.fft.irfft2(np.fft.rfft2(image, s=shape) *
                          np.fft.rfft2(kernel[::-1, ::-1], s=shape), s=shape)
    return result[kh - 1:h, kw - 1:w].copy()


def _center(placement, templates):
    t = templates[placement.template]
    return placement.left + t.anchor_x, placement.top + t.anchor_y


def _patch(array, placement, templates):
    h, w = templates[placement.template].ink.shape
    return array[placement.top:placement.top + h, placement.left:placement.left + w]


def _quality(placement, templates, ink, weight):
    t = templates[placement.template].ink
    q, weights = _patch(ink, placement, templates), _patch(weight, placement, templates)
    mass = float(np.sum(weights * t))
    numerator = float(np.sum(weights * q * t))
    denominator = float(np.sqrt(np.sum(weights * q*q) * np.sum(weights * t*t)))
    score = numerator / denominator if denominator > 1e-12 else 0.
    support = numerator / mass if mass > 1e-12 else 0.
    mse = float(np.sum(weights * (q - t)**2) / max(weights.sum(), 1))
    return score, support, mass / t.sum(), mse


def extract_markers(
    analysis, template, *, color=None, scales=(1.,), anchor=None,
    color_tolerance: float = .2, min_score: float = .65,
    min_color_support: float = .65, min_visible_fraction: float = .6,
    min_distance: float | None = None, min_gain: float = .12,
    frame_margin: float = 1., exclude_regions=(), max_candidates: int = 500,
    refinement_steps: int = 2, tick_sigma_px: float = 1 / math.sqrt(12),
) -> MarkerPointCloud:
    """Locate reference-marker hypotheses and reuse pre_analyzer calibration.

    First screen by marker-color support and masked normalized correlation.
    Foreign colors (e.g. red over a black marker) are masked as unknown. Fit an
    opacity union of proposed markers by greedy weighted squared-error reduction
    and bounded coordinate refinement; additional markers must explain additional
    image structure. This permits overlap without requiring a white patch around
    every marker. Scores are not probabilities; strongly occluded/merged markers
    can remain ambiguous. Identical legend symbols require context or exclusions.

    template accepts an RGB/grayscale/RGBA array or an image path. Its perimeter
    estimates the background; the reference should contain one clean marker with
    background margins. The default anchor is its ink centroid, independent of
    crop padding. anchor can override it in original template pixel coordinates.
    scales accounts for reference/data size differences. Search is bounded by
    max_candidates and refinement_steps; no additional image library is needed.
    """
    tolerance = _positive_number(color_tolerance, "color_tolerance")
    if tolerance > math.sqrt(3):
        raise ValueError("color_tolerance must not exceed sqrt(3).")
    thresholds = {}
    for name, value in (("min_score", min_score), ("min_color_support", min_color_support),
                        ("min_visible_fraction", min_visible_fraction), ("min_gain", min_gain)):
        thresholds[name] = _positive_number(value, name)
        if thresholds[name] > 1:
            raise ValueError(f"{name} must be in (0, 1].")
    min_score, min_color_support = thresholds["min_score"], thresholds["min_color_support"]
    min_visible_fraction, min_gain = thresholds["min_visible_fraction"], thresholds["min_gain"]
    margin = _positive_number(frame_margin, "frame_margin", allow_zero=True)
    scale_values = np.asarray(scales, dtype=float)
    if (scale_values.ndim != 1 or not 1 <= len(scale_values) <= 12 or not np.isfinite(scale_values).all()
            or np.any((scale_values < .25) | (scale_values > 4))):
        raise ValueError("Supply 1 to 12 template scales in [0.25, 4].")
    for name, value, minimum, maximum in (("max_candidates", max_candidates, 1, 10000),
                                          ("refinement_steps", refinement_steps, 0, 5)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    exclusions = tuple(exclude_regions)
    if any(not isinstance(region, Rectangle) for region in exclusions):
        raise ValueError("exclude_regions must contain Rectangles in source pixel coordinates.")
    rgb, frame, ticks = _analysis_parts(analysis)
    calibration = calibrate_axes(ticks, tick_sigma_px=tick_sigma_px)
    templates, background, target, template_shape = _prepare(template, color, tolerance, scale_values, anchor)
    distance = .45 * min(t.diameter for t in templates) if min_distance is None else _positive_number(min_distance, "min_distance")
    if 2 * margin >= min(frame.rectangle.width, frame.rectangle.height):
        raise ValueError("frame_margin leaves no plot interior.")
    ink, weight = _project(rgb, background, target, tolerance)
    r = frame.rectangle

    def inside(x, y):
        result = (x > r.left + margin) & (x < r.right - margin) & (y > r.top + margin) & (y < r.bottom - margin)
        for region in exclusions:
            result &= np.logical_not((x >= region.left) & (x <= region.right) & (y >= region.top) & (y <= region.bottom))
        return result

    candidates, notes, usable_scales, truncated = [], list(calibration.notes), 0, False
    for index, t in enumerate(templates):
        if any(a > b for a, b in zip(t.ink.shape, ink.shape)):
            continue
        usable_scales += 1
        ones = np.ones(t.ink.shape)
        numerator = _correlate(weight * ink, t.ink)
        energy = np.maximum(_correlate(weight * ink*ink, ones) * _correlate(weight, t.ink*t.ink), 0)
        score = np.divide(numerator, np.sqrt(energy), out=np.zeros_like(numerator), where=energy > 1e-12)
        mass = _correlate(weight, t.ink)
        support = np.divide(numerator, mass, out=np.zeros_like(numerator), where=mass > 1e-12)
        y, x = np.indices(score.shape)
        valid = ((score >= min_score) & (support >= min_color_support) & (mass >= min_visible_fraction * t.ink.sum())
                 & inside(x + t.anchor_x, y + t.anchor_y))
        metric = np.where(valid, score, -np.inf)
        padded = np.pad(metric, 2, constant_values=-np.inf)
        h, w = score.shape
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx or dy:
                    valid &= metric >= padded[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]
        yy, xx = np.nonzero(valid)
        truncated |= len(yy) > max_candidates
        order = np.argsort(-score[yy, xx], kind="stable")[:max_candidates]
        candidates.extend(_Placement(int(yy[i]), int(xx[i]), index, float(score[yy[i], xx[i]])) for i in order)
    if not usable_scales:
        raise ValueError("All scaled templates are larger than the image.")
    candidates.sort(key=lambda c: -c.score)
    if truncated or len(candidates) > max_candidates:
        notes.append(f"Candidate search capped at {max_candidates}; some hypotheses were not fitted.")
    candidates = candidates[:max_candidates]
    candidate_count = len(candidates)
    canvas = np.zeros(ink.shape)
    log_ink = [np.log1p(-np.minimum(t.ink, 1 - 1e-6)) for t in templates]

    def gain(p):
        remaining = np.exp(_patch(canvas, p, templates))
        increment = remaining * templates[p.template].ink
        return float(np.sum(_patch(weight, p, templates) * (
            2 * (_patch(ink, p, templates) - 1 + remaining) * increment - increment*increment)))

    selected = []
    while candidates:
        gains = [gain(p) - min_gain * templates[p.template].energy for p in candidates]
        best = int(np.argmax(gains))
        p = candidates.pop(best)
        if gains[best] <= 0:
            break
        selected.append(p)
        _patch(canvas, p, templates)[:] += log_ink[p.template]
        px, py = _center(p, templates)
        candidates = [c for c in candidates if math.dist((px, py), _center(c, templates)) >= distance]
    for _ in range(refinement_steps):
        refined = []
        for i, p in enumerate(selected):
            _patch(canvas, p, templates)[:] -= log_ink[p.template]
            cx, cy = _center(p, templates)
            others = [*refined, *selected[i + 1:]]
            best, best_value = None, 0.
            for index, t in enumerate(templates):
                top, left = round(cy - t.anchor_y), round(cx - t.anchor_x)
                for y in range(top - 2, top + 3):
                    for x in range(left - 2, left + 3):
                        if y < 0 or x < 0 or y + t.ink.shape[0] > ink.shape[0] or x + t.ink.shape[1] > ink.shape[1]:
                            continue
                        proposal = _Placement(y, x, index, 0.)
                        center = _center(proposal, templates)
                        if not inside(*center) or any(math.dist(center, _center(other, templates)) < distance for other in others):
                            continue
                        quality = _quality(proposal, templates, ink, weight)
                        if quality[0] < min_score or quality[1] < min_color_support or quality[2] < min_visible_fraction:
                            continue
                        value = gain(proposal) - min_gain * t.energy
                        if value > best_value:
                            best, best_value = proposal, value
            if best is not None:
                refined.append(best)
                _patch(canvas, best, templates)[:] += log_ink[best.template]
        selected = refined
    matches = []
    for p in selected:
        t = templates[p.template]
        px, py = _center(p, templates)
        score, support, visible, mse = _quality(p, templates, ink, weight)
        _patch(canvas, p, templates)[:] -= log_ink[p.template]
        unique_gain = gain(p) / t.energy
        _patch(canvas, p, templates)[:] += log_ink[p.template]
        overlap = any(other is not p and math.dist((px, py), _center(other, templates)) < .5 * (
            t.diameter + templates[other.template].diameter) for other in selected)
        matches.append(MarkerMatch(px, py, t.scale, score, support, visible, mse, unique_gain, overlap))
    matches.sort(key=lambda m: (m.pixel_x, m.pixel_y))
    pixels = np.array([(m.pixel_x, m.pixel_y) for m in matches], dtype=float).reshape(-1, 2)
    x, eps_x = calibration.x.transform(pixels[:, 0])
    y, eps_y = calibration.y.transform(pixels[:, 1])
    notes.append("Marker hypotheses may include identical legend symbols; exclude_regions can remove known annotation areas.")
    notes.append("eps excludes marker-center localization error; overlapping/occluded hypotheses require visual checking.")
    if exclusions:
        notes.append(f"Applied {len(exclusions)} explicitly supplied exclusion regions.")
    return MarkerPointCloud(x, y, np.column_stack((eps_x, eps_y)), tuple(matches),
                            tuple(float(v) for v in target), frame, calibration, rgb.shape[:2],
                            template_shape, candidate_count, exclusions, tuple(notes))
