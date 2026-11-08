"""Conservative consistency checks for observed tick positions and axis values."""

from dataclasses import dataclass, replace
from itertools import combinations
import math

import numpy as np
from numpy.typing import ArrayLike

from .ticks import AxisTickValidation, TickGridDetection, TickOutlier


def _axis_options(values, linear, logarithmic, axis):
    for flag in (linear, logarithmic):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{axis}_lin and {axis}_log must be booleans.")
    if linear and logarithmic:
        raise ValueError(f"{axis}_lin and {axis}_log cannot both be True.")
    scale = "log" if logarithmic else "linear" if linear or values is not None else None
    if values is None:
        return scale, None
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{axis}_ticks must be a one-dimensional numeric array.")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{axis}_ticks must be finite.")
    if len(array) > 1 and not (np.all(np.diff(array) > 0) or np.all(np.diff(array) < 0)):
        raise ValueError(f"{axis}_ticks must be strictly increasing or decreasing.")
    if scale == "log" and np.any(array <= 0):
        raise ValueError(f"{axis}_ticks must be positive on a logarithmic axis.")
    return scale, tuple(float(value) for value in array)


def _count_option(count, references, axis):
    if count is not None and (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer)) or count < 0
    ):
        raise ValueError(f"expected_{axis}_ticks must be a nonnegative integer or None.")
    if references is not None:
        if count is not None and count != len(references):
            raise ValueError(f"expected_{axis}_ticks conflicts with the number of {axis}_ticks values.")
        return len(references)
    return count


@dataclass(frozen=True)
class _Model:
    slope: float
    offset: float
    pairs: tuple[tuple[int, int], ...]
    rms: float


def _fit_line(u, pixels):
    centered = u - u.mean()
    denominator = np.dot(centered, centered)
    if denominator <= 0:
        return None
    slope = float(np.dot(centered, pixels - pixels.mean()) / denominator)
    return slope, float(pixels.mean() - slope * u.mean())


def _match(predicted, observed, tolerance):
    """Nearest one-to-one, ordered references; never manufacture observations."""
    right = np.clip(np.searchsorted(predicted, observed), 0, len(predicted) - 1)
    left = np.maximum(right - 1, 0)
    indexes = np.where(abs(predicted[left] - observed) <= abs(predicted[right] - observed), left, right)
    errors = abs(predicted[indexes] - observed)
    used = set()
    pairs = []
    for observation in np.argsort(errors, kind="stable"):
        reference = int(indexes[observation])
        if errors[observation] <= tolerance and reference not in used:
            pairs.append((reference, int(observation)))
            used.add(reference)
    return tuple(sorted(pairs))


def _refine(u, observed, slope, offset, tolerance):
    for _ in range(3):
        pairs = _match(slope * u + offset, observed, tolerance)
        if len(pairs) < 3:
            return None
        refs, obs = np.array(pairs).T
        fitted = _fit_line(u[refs], observed[obs])
        if fitted is None or fitted[0] <= 0:
            return None
        slope, offset = fitted
    pairs = _match(slope * u + offset, observed, tolerance)
    if len(pairs) < 3:
        return None
    refs, obs = np.array(pairs).T
    rms = float(np.sqrt(np.mean((slope * u[refs] + offset - observed[obs]) ** 2)))
    return _Model(slope, offset, pairs, rms)


def _reference_models(u, observed, tolerance):
    # Bounded anchor sets keep hypotheses manageable without parallel workers.
    refs = np.unique(np.linspace(0, len(u) - 1, min(len(u), 16)).round().astype(int))
    obs = np.unique(np.linspace(0, len(observed) - 1, min(len(observed), 16)).round().astype(int))
    models = {}
    for i, j in combinations(refs, 2):
        if u[j] - u[i] < 0.5:
            continue
        for first, last in combinations(obs, 2):
            if observed[last] - observed[first] < 2 * tolerance:
                continue
            slope = float((observed[last] - observed[first]) / (u[j] - u[i]))
            model = _refine(u, observed, slope, float(observed[first] - slope * u[i]), tolerance)
            if model is not None and (
                model.pairs not in models or model.rms < models[model.pairs].rms
            ):
                models[model.pairs] = model
    return list(models.values())


def _best_models(models, minimum, tolerance):
    if not models:
        return []
    count = max(len(model.pairs) for model in models)
    if count < minimum:
        return []
    models = [model for model in models if len(model.pairs) == count]
    best_rms = min(model.rms for model in models)
    # Subpixel fit differences are not enough to resolve competing mappings.
    return [model for model in models if model.rms <= best_rms + 0.25 * tolerance]


def _manual_check(ticks, references, scale, axis, tolerance):
    validation = AxisTickValidation(scale, "insufficient", references, tolerance)
    if not references:
        rejected = tuple(TickOutlier(tick, "not_in_manual_values") for tick in ticks)
        return (), rejected, replace(validation, status="empty_reference"), f"{axis}: empty manual tick list; separated {len(ticks)} observations."
    if not ticks:
        return ticks, (), replace(validation, status="no_observations", missing_values=references), f"{axis}: no observed ticks for the supplied values."
    ordered = np.array(references[::-1] if axis == "y" else references)
    z = np.log(ordered) if scale == "log" else ordered
    normalizer = max(float(np.max(abs(z))), 1.0)
    scaled = z / normalizer
    observed = np.array([tick.position for tick in ticks])
    if len(references) == 1:
        if len(ticks) == 1:
            ticks = (replace(ticks[0], value=references[0]),)
        return ticks, (), validation, f"{axis}: one reference value cannot check a scale or isolate outliers."
    u = (scaled - scaled[0]) / (scaled[-1] - scaled[0])
    if len(references) == len(ticks) == 2:
        model = _Model(float(observed[1] - observed[0]), float(observed[0]), ((0, 0), (1, 1)), 0)
        models = [model]
    else:
        minimum = max(3, math.ceil(len(references) / 2))
        if min(len(ticks), len(references)) < minimum:
            return ticks, (), validation, f"{axis}: insufficient independent anchors; no ticks separated."
        models = _best_models(_reference_models(u, observed, tolerance), minimum, tolerance)
    if not models:
        return ticks, (), validation, f"{axis}: no sufficiently supported {scale} mapping; no ticks separated."
    observation_sets = {tuple(sorted(obs for _, obs in model.pairs)) for model in models}
    if len(observation_sets) != 1:
        return ticks, (), replace(validation, status="ambiguous"), f"{axis}: competing mappings retain different ticks; no observations separated."
    model = min(models, key=lambda item: item.rms)
    inliers = set(next(iter(observation_sets)))
    unique_mapping = len({model.pairs for model in models}) == 1
    values_by_observation = {obs: float(ordered[ref]) for ref, obs in model.pairs}
    accepted = tuple(replace(tick, value=values_by_observation[index] if unique_mapping else None)
                     for index, tick in enumerate(ticks) if index in inliers)
    rejected = []
    for index, tick in enumerate(ticks):
        if index not in inliers:
            residual = min(float(np.min(abs(candidate.slope * u + candidate.offset - tick.position)))
                           for candidate in models)
            rejected.append(TickOutlier(tick, "duplicate_reference" if residual <= tolerance else "off_reference", residual))
    if not unique_mapping:
        return accepted, tuple(rejected), replace(validation, status="ambiguous_values"), f"{axis}: inlier set agrees, but value assignment is ambiguous; separated {len(rejected)} ticks."
    matched_refs = {ref for ref, _ in model.pairs}
    missing = tuple(value for value in references if value not in {float(ordered[index]) for index in matched_refs})
    slope_scaled = model.slope / (scaled[-1] - scaled[0])
    validation = replace(
        validation, status="two_point" if len(references) == 2 else "matched",
        missing_values=missing, pixel_slope=float(slope_scaled / normalizer),
        pixel_offset=float(model.offset - slope_scaled * scaled[0]), rms_error_pixels=model.rms,
    )
    note = f"{axis}: {len(accepted)} matched, {len(rejected)} separated, {len(missing)} missing ({scale})."
    if validation.status == "two_point":
        note += " Two anchors give no independent outlier test."
    return accepted, tuple(rejected), validation, note


def _spacing_check(ticks, tolerance, axis):
    validation = AxisTickValidation("linear", "insufficient", None, tolerance)
    if len(ticks) < 4:
        return ticks, (), validation, f"{axis}: at least four ticks needed for a spacing-only check."
    observed = np.array([tick.position for tick in ticks])
    candidates = np.diff(observed)
    candidates = np.unique(candidates[candidates >= 4 * tolerance])
    models = {}
    for step in candidates:
        for origin in observed:
            slope, offset = float(step), float(origin)
            for _ in range(3):
                indexes = np.rint((observed - offset) / slope)
                good = np.flatnonzero(abs(observed - (offset + slope * indexes)) <= tolerance)
                if len(good) < 3 or len(np.unique(indexes[good])) != len(good):
                    break
                fitted = _fit_line(indexes[good], observed[good])
                if fitted is None or fitted[0] <= 0:
                    break
                slope, offset = fitted
            else:
                indexes = np.rint((observed - offset) / slope)
                good = np.flatnonzero(abs(observed - (offset + slope * indexes)) <= tolerance)
                if len(good) < 3 or len(np.unique(indexes[good])) != len(good):
                    continue
                pairs = tuple((int(indexes[index]), int(index)) for index in good)
                rms = float(np.sqrt(np.mean((observed[good] - (offset + slope * indexes[good])) ** 2)))
                key = tuple(int(index) for index in good)
                if key not in models or rms < models[key].rms:
                    models[key] = _Model(slope, offset, pairs, rms)
    best = _best_models(list(models.values()), max(4, math.ceil(0.6 * len(ticks))), tolerance)
    if not best:
        return ticks, (), validation, f"{axis}: no dominant regular spacing; no ticks separated."
    sets = {tuple(obs for _, obs in model.pairs) for model in best}
    if len(sets) != 1:
        return ticks, (), replace(validation, status="ambiguous"), f"{axis}: regular-spacing models disagree; no ticks separated."
    model = min(best, key=lambda item: item.rms)
    inliers = set(next(iter(sets)))
    accepted = tuple(tick for index, tick in enumerate(ticks) if index in inliers)
    rejected = tuple(TickOutlier(tick, "off_spacing", float(abs(
        tick.position - (model.offset + model.slope * round((tick.position - model.offset) / model.slope))
    ))) for index, tick in enumerate(ticks) if index not in inliers)
    return accepted, rejected, replace(validation, status="spacing_only", spacing_pixels=model.slope,
                                       rms_error_pixels=model.rms), f"{axis}: regular pixel-spacing check; {len(rejected)} ticks separated. Numeric values are not calibrated."


def check_tick_consistency(
    detection: TickGridDetection | None, *,
    x_ticks: ArrayLike | None = None, y_ticks: ArrayLike | None = None,
    x_lin: bool = False, y_lin: bool = False, x_log: bool = False, y_log: bool = False,
    tolerance_px: float = 2.0,
    expected_x_ticks: int | None = None, expected_y_ticks: int | None = None,
) -> TickGridDetection | None:
    """Match manual axis values to observations and retain separated evidence.

    x_ticks are ordered left to right; y_ticks bottom to top, including reversed
    axes. Values must be finite and strictly monotonic; logarithmic values must
    be positive. Manual arrays default to linear unless *_log=True. With no
    options, preserve the original detection. Linear-only checks use a dominant
    regular-spacing heuristic; log-only checks need numeric references and do
    not reject arbitrary major/minor patterns.

    Fit pixels = slope * value + offset (or log(value)) independently of frame
    limits, requiring >=3 anchors and >=half the references for an outlier test.
    Equally plausible competing inlier sets leave observations untouched.
    Unmatched observations can be valid minor ticks omitted from a manual list.
    """
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("tolerance_px must be finite and positive.")
    options = {
        "x": _axis_options(x_ticks, x_lin, x_log, "x"),
        "y": _axis_options(y_ticks, y_lin, y_log, "y"),
    }
    counts = {}
    for axis, requested in (("x", expected_x_ticks), ("y", expected_y_ticks)):
        values = options[axis][1]
        if requested is None and values is None and detection is not None:
            requested = getattr(detection, f"expected_{axis}_ticks")
        counts[axis] = _count_option(requested, values, axis)
    if detection is None:
        return None
    if not any(scale is not None for scale, _ in options.values()):
        return detection
    changes = {}
    notes = list(detection.notes)
    for axis, (scale, values) in options.items():
        if scale is None:
            continue
        raw = (*getattr(detection, f"{axis}_ticks"),
               *(outlier.tick for outlier in getattr(detection, f"rejected_{axis}_ticks")))
        raw = tuple(replace(tick, value=None) for tick in sorted(raw, key=lambda tick: tick.position))
        if values is not None:
            accepted, rejected, validation, note = _manual_check(raw, values, scale, axis, tolerance_px)
        elif scale == "linear":
            accepted, rejected, validation, note = _spacing_check(raw, tolerance_px, axis)
        else:
            accepted, rejected = raw, ()
            validation = AxisTickValidation("log", "needs_values", None, tolerance_px)
            note = f"{axis}: logarithmic spacing needs manual values; major/minor patterns retained."
        changes[f"{axis}_ticks"] = accepted
        changes[f"rejected_{axis}_ticks"] = rejected
        changes[f"{axis}_validation"] = validation
        changes[f"expected_{axis}_ticks"] = counts[axis]
        notes.append(note)
    return replace(detection, **changes, notes=tuple(notes))
