"""Color-selected pixel clouds calibrated using already validated tick values."""

from dataclasses import dataclass
import math

from matplotlib.colors import to_rgb
import numpy as np
from numpy.typing import ArrayLike

from .frame import FrameDetection
from .preprocessing import _normalized_image
from .ticks import TickGridDetection


@dataclass(frozen=True)
class AxisCalibration:
    """pixels = slope * u + offset, where u is value or natural log(value)."""

    scale: str
    pixel_slope: float
    pixel_offset: float
    anchor_count: int
    mean_u: float
    sum_squared_u: float
    residual_sigma_pixels: float
    tick_sigma_pixels: float

    def transform(self, pixels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return axis values and approximate 1-sigma calibration uncertainty.

        Propagate slope/intercept covariance, conditional on correct tick-value
        assignments and exact reference values. Tick errors are independent and
        homoscedastic. The query pixel itself is held fixed: curve width, source
        measurement errors and query-pixel quantization are not included.
        """
        pixels = np.asarray(pixels, dtype=float)
        if not np.isfinite(pixels).all():
            raise ValueError("Pixel positions must be finite.")
        u = (pixels - self.pixel_offset) / self.pixel_slope
        sigma = max(self.residual_sigma_pixels, self.tick_sigma_pixels)
        eps_u = sigma / abs(self.pixel_slope) * np.sqrt(
            1 / self.anchor_count + (u - self.mean_u) ** 2 / self.sum_squared_u
        )
        with np.errstate(over="raise", invalid="raise"):
            try:
                value = np.exp(u) if self.scale == "log" else u
                eps = value * eps_u if self.scale == "log" else eps_u
            except FloatingPointError as error:
                raise ValueError("Axis transformation exceeds finite floating-point range.") from error
        return value, eps


@dataclass(frozen=True)
class CoordinateCalibration:
    x: AxisCalibration
    y: AxisCalibration
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ColorPointCloud:
    """One point per selected pixel, with axis-wise calibration uncertainties.

    Unpack as x, y, eps; eps has shape (N, 2), ordered eps_x, eps_y. pixels
    retains original (column, row) positions. No centerline or experimental
    measurement points are inferred from the thickness of the rendered stroke.
    """

    x: np.ndarray
    y: np.ndarray
    eps: np.ndarray
    pixels: np.ndarray
    color: tuple[float, float, float]
    color_tolerance: float
    frame: FrameDetection
    calibration: CoordinateCalibration
    image_shape: tuple[int, int]
    matched_pixel_count: int
    component_count: int
    notes: tuple[str, ...]

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.eps

    def to_array(self) -> np.ndarray:
        """Columns: x, y, eps_x, eps_y, pixel_x, pixel_y; row-major pixel order."""
        return np.column_stack((self.x, self.y, self.eps, self.pixels))


def _positive_number(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number.")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
    return value


def calibrate_axes(ticks: TickGridDetection, *, tick_sigma_px: float = 1 / math.sqrt(12)) -> CoordinateCalibration:
    """Reuse unique numerical mappings from check_tick_consistency/pre_analyzer.

    No frame endpoints or missing labels are invented as calibration anchors.
    Use accepted, value-assigned ticks only, including two-point mappings. The
    residual variance SSE/(n-2) has a tick-localization floor tick_sigma_px**2;
    the default floor corresponds to a uniformly located tick within +/-0.5 px.
    A two-anchor calibration uses that floor without a residual variance estimate.
    """
    tick_sigma_px = _positive_number(tick_sigma_px, "tick_sigma_px")
    if not isinstance(ticks, TickGridDetection):
        raise ValueError("A successful tick analysis with numerical axis values is required.")
    if ticks.frame.color_check is not None and ticks.frame.color_check.status in ("suspect", "rejected"):
        raise ValueError("Cannot calibrate an unresolved frame.")
    fitted, notes = {}, []
    for axis in ("x", "y"):
        validation = getattr(ticks, f"{axis}_validation")
        if validation is None or validation.status not in ("matched", "two_point"):
            raise ValueError(f"{axis}: a unique numerical tick mapping is required; supply axis values to pre_analyzer.")
        anchors = getattr(ticks, f"{axis}_ticks")
        if len(anchors) < 2 or any(tick.value is None for tick in anchors):
            raise ValueError(f"{axis}: at least two value-assigned ticks are required.")
        values = np.array([tick.value for tick in anchors], dtype=float)
        pixels = np.array([tick.position for tick in anchors], dtype=float)
        slope, offset = validation.pixel_slope, validation.pixel_offset
        if (validation.scale not in ("linear", "log") or slope is None or offset is None
                or not np.isfinite([slope, offset]).all() or slope == 0
                or not np.isfinite(values).all() or not np.isfinite(pixels).all()
                or (validation.scale == "log" and np.any(values <= 0))):
            raise ValueError(f"{axis}: invalid numerical calibration.")
        u = np.log(values) if validation.scale == "log" else values
        mean = float(u.mean())
        squared = float(np.sum((u - mean) ** 2))
        if not math.isfinite(squared) or squared <= 0:
            raise ValueError(f"{axis}: tick values do not span a calibratable range.")
        residual = pixels - (slope * u + offset)
        sigma = float(np.sqrt(np.dot(residual, residual) / (len(u) - 2))) if len(u) > 2 else 0.
        fitted[axis] = AxisCalibration(validation.scale, slope, offset, len(u), mean, squared, sigma, tick_sigma_px)
        if len(u) == 2:
            notes.append(f"{axis}: two anchors; uncertainty uses the tick-localization floor only.")
    notes.append("eps is axis-wise approximate 1-sigma calibration uncertainty; stroke width and experimental errors are excluded.")
    notes.append("Calibration errors are shared across points; eps entries are marginal, not independent point errors.")
    return CoordinateCalibration(fitted["x"], fitted["y"], tuple(notes))


def _color_rgb(color):
    if isinstance(color, str):
        return np.array(to_rgb(color))
    source = np.asarray(color)
    if source.shape != (3,) or not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
        raise ValueError("color must be a color name/hex string or three RGB channels.")
    values = source.astype(float)
    if np.issubdtype(source.dtype, np.integer):
        values /= 255
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Use integer RGB in [0, 255] or floating RGB in [0, 1].")
    return values


def _analysis_parts(analysis):
    if not isinstance(analysis, (tuple, list)) or len(analysis) != 4:
        raise ValueError("analysis must be the (image, frame, ticks, colors) result from pre_analyzer.")
    image, frame, ticks, _ = analysis
    if not isinstance(frame, FrameDetection) or not isinstance(ticks, TickGridDetection):
        raise ValueError("A successful frame and tick analysis is required.")
    if frame.rectangle != ticks.frame.rectangle:
        raise ValueError("Frame and ticks describe different rectangles.")
    source = np.asarray(image)
    if np.issubdtype(source.dtype, np.integer):
        if not source.size or source.min() < 0 or source.max() > 255:
            raise ValueError("Integer image channels must be in [0, 255].")
        source = source.astype(float) / 255
    rgb = _normalized_image(source)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    height, width = rgb.shape[:2]
    r = frame.rectangle
    if not (0 <= r.left < r.right < width and 0 <= r.top < r.bottom < height):
        raise ValueError("Frame bounds lie outside this image.")
    if frame.color_check is not None and frame.color_check.status in ("suspect", "rejected"):
        raise ValueError("Cannot extract from an unresolved frame.")
    return rgb, frame, ticks


def _components(mask):
    """8-connected components via row runs, without another image dependency."""
    parent, sizes, spans, previous = [], [], [], []

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for row in mask:
        columns = np.flatnonzero(row)
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1) if len(columns) else []
        current, first = [], 0
        for group in groups:
            left, right = int(group[0]), int(group[-1])
            index = len(parent)
            parent.append(index)
            sizes.append(right - left + 1)
            spans.append((left, right))
            while first < len(previous) and previous[first][1] < left - 1:
                first += 1
            for prev_left, prev_right, prev_id in previous[first:]:
                if prev_left > right + 1:
                    break
                a, b = root(index), root(prev_id)
                if a != b:
                    parent[b] = a
                    sizes[a] += sizes[b]
            current.append((left, right, index))
        previous = current
    labels = np.concatenate([np.full(right - left + 1, root(i), dtype=int)
                             for i, (left, right) in enumerate(spans)]) if spans else np.empty(0, dtype=int)
    roots = sorted({root(i) for i in range(len(parent))})
    largest = max(roots, key=lambda i: sizes[i]) if roots else None
    return labels, len(roots), largest


def extract_color_data(
    analysis, color, *, color_tolerance: float = 0.2, components: str = "all",
    frame_margin: float = 4., tick_sigma_px: float = 1 / math.sqrt(12),
) -> ColorPointCloud:
    """Select original-RGB pixels inside an analyzed frame and calibrate them.

    analysis is the unmodified four-part pre_analyzer result. Both axes need
    unique numeric tick-value assignments. Color distance is Euclidean in
    normalized RGB; matching includes nearby antialiasing shades within the
    tolerance. color accepts a Matplotlib color name/hex, integer RGB [0,255],
    or float RGB [0,1]. RGBA inputs are composited over white.

    Keep all matches by default, including disconnected markers. For a connected
    curve use components='largest' (8-connectivity, greatest pixel area) to remove
    smaller legend strokes. This is a geometric heuristic, not semantic legend
    recognition; same-color touching objects remain inseparable. frame_margin
    excludes border strips in pixels. No centerline, resampling or interpolation
    is applied. eps includes calibration covariance only, separately for x/y.
    """
    tolerance = _positive_number(color_tolerance, "color_tolerance", allow_zero=True)
    margin = _positive_number(frame_margin, "frame_margin", allow_zero=True)
    if tolerance > math.sqrt(3):
        raise ValueError("color_tolerance must not exceed the RGB cube diagonal sqrt(3).")
    if components not in ("all", "largest"):
        raise ValueError("components must be 'all' or 'largest'.")
    rgb, frame, ticks = _analysis_parts(analysis)
    calibration = calibrate_axes(ticks, tick_sigma_px=tick_sigma_px)
    target = _color_rgb(color)
    r = frame.rectangle
    if 2 * margin >= min(r.width, r.height):
        raise ValueError("frame_margin leaves no plot interior.")
    mask = np.sum((rgb - target) ** 2, axis=2) <= tolerance ** 2
    columns, rows = np.arange(rgb.shape[1]), np.arange(rgb.shape[0])
    mask &= ((columns > r.left + margin) & (columns < r.right - margin))[None, :]
    mask &= ((rows > r.top + margin) & (rows < r.bottom - margin))[:, None]
    pixel_y, pixel_x = np.nonzero(mask)
    matched = len(pixel_x)
    labels, count, largest = _components(mask)
    notes = list(calibration.notes)
    if components == "largest" and largest is not None:
        keep = labels == largest
        pixel_x, pixel_y = pixel_x[keep], pixel_y[keep]
        if count > 1:
            notes.append(f"Selected largest of {count} color components; excluded {matched - len(pixel_x)} pixels.")
    if not matched:
        notes.append("No pixels match the requested color inside the frame margin.")
    x, eps_x = calibration.x.transform(pixel_x)
    y, eps_y = calibration.y.transform(pixel_y)
    return ColorPointCloud(x, y, np.column_stack((eps_x, eps_y)),
                           np.column_stack((pixel_x, pixel_y)), tuple(float(v) for v in target),
                           tolerance, frame, calibration, rgb.shape[:2], matched, count, tuple(notes))
