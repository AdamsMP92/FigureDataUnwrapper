# FigureDataUnwrapper

A Python package for reconstructing data from scientific figures when the
original datasets are unavailable. The current step detects a rectangular plot
frame in image pixel coordinates, including frames with missing sides, then
identifies tick marks and grid lines relative to that frame.

## Getting started

With Pixi installed, run from the repository root:

```sh
pixi install
pixi run example
pixi run detect-frame
pixi run detect-ticks
pixi run binarization-trial
pixi run test
```

The default example generates 13 points on the parabola `y = x**2`, with `x`
evenly spaced between -3 and 3, saves `examples/parabola.png`, and runs the
`pre_analyzer` wrapper from `examples/parabola_check.py`. Its standard plausibility
plot is saved as `examples/parabola_check.png`, without opening a window.
Use `pixi run generate-example` to generate just the reference input image.
`detect-frame` reads only the rendered PNG and writes `parabola_mask.png`
(black foreground on white background) and `parabola_frame.png` (red overlay).
Extraction has no access to renderer geometry or the original data.

## Standard plausibility check

`pre_analyzer` bundles image loading, frame/tick/grid extraction, color ranking,
and the PNG plausibility plot in one call:

```python
from pathlib import Path
from figure_data_unwrapper import pre_analyzer

image, frame, ticks, colors = pre_analyzer(Path("examples/parabola.png"), n_colors=7)
```

The image returned by the wrapper is normalized to 0–1, including integer JPEG
input. It preserves the four-part return tuple and writes `<stem>_check.png`
beside the input by default. Optional `output`, `expected_x_ticks`,
`expected_y_ticks`, and `show_ticks` keywords customize the destination, count
hints, and overlay. If geometric frame reconstruction fails, frame and ticks are
None. Unresolved RGB checks retain the candidate frame and its diagnostics but
skip tick/grid extraction. Color ranking and the diagnostic PNG are still produced.

The shared exporter is `plot_check` in `src/figure_data_unwrapper/plotting.py`.
The original occupies the upper-left region at its native pixel resolution.
Observed frame sides are solid red, inferred sides dashed red, and grid
candidates dashed cyan. Detected ticks are shown by default as orange segments
with circles at their frame intersections. Segment directions and lengths follow
the measured inward/outward branches. Set `show_ticks=False` to hide them.
An information panel to the right shows the legend, tick/grid counts, up to ten
supplied color-ranking entries, and analysis notes. The panel adds canvas space
without rescaling or covering the input image. RGB values in the ranking remain
unchanged; swatches use rounded hex labels for display.

`plot_check` renders supplied results without rerunning extraction and returns
the output Path. It accepts normalized grayscale/RGB/RGBA and integer channels
in 0–255, composites alpha over white, and renders boolean foreground masks
black on white. Mismatched frame/tick rectangles or out-of-image bounds raise
an error. If no frame is available, it still saves the image with a status note,
so failed extraction can also be inspected.

`pixi run example` and its alias `pixi run check-parabola` create this plot.
The configurable `detect-frame` and `detect-ticks` examples also call the shared
function; their plotting code no longer needs to be duplicated.

## Manual tick values and scale consistency

```python
import numpy as np
from figure_data_unwrapper import pre_analyzer

image, frame, ticks, colors = pre_analyzer(
    "examples/real_example1.png", n_colors=7,
    x_ticks=np.linspace(0, 360, 17),
    y_ticks=[0, 2.5, 5, 7.5, 10],
    x_lin=True, y_lin=True,
)
```

The arrays contain **axis values**, not pixels: x in left-to-right order, y in
bottom-to-top order. Strictly decreasing arrays support reversed axes. Values
must be finite, distinct and strictly monotonic. Select `x_lin`/`x_log` and
`y_lin`/`y_log` independently; both flags for the same axis are invalid.
Manual arrays default to linear if neither flag is selected. Logarithmic values
must be positive. A count hint supplied alongside values must match their count.

Consistency fits `pixel = a * value + b`, or `pixel = a * log(value) + b` (natural logarithm) for
log axes. Frame bounds are not assumed to equal the smallest/largest supplied
values. Observed positions are preserved, and the fitted mapping assigns
`AxisTick.value`. The default tolerance is `tick_tolerance_px=2.0` in image pixels.
At least three independent anchors and half of the reference values must match
for an outlier check. Two anchors can define a mapping but cannot independently
establish which additional observations are wrong.

Accepted observations remain in `ticks.x_ticks`/`ticks.y_ticks`. Separated ones
are retained in `ticks.rejected_x_ticks`/`ticks.rejected_y_ticks`; each TickOutlier
contains the original `tick`, a reason, and a residual in pixels when available.
`x_validation`/`y_validation` expose the scale, status, missing reference values,
fit error, and mapping coefficients. Missing reference values are reported for
a unique mapping. Orange marks are accepted; purple dotted segments and crosses
show separated evidence. The check panel reports both counts.

Competing fits with different inlier sets leave the observations untouched
(`ambiguous`). If the inlier set agrees but assigning values is ambiguous,
unsupported observations can be separated while values remain unassigned
(`ambiguous_values`). Missing, ambiguous, and insufficient cases are also
described in `ticks.notes`. Separation is a consistency decision: valid minor
ticks omitted from a supplied reference list can also be unmatched, so it is
not a claim that every separated image stroke is false.

Without manual arrays, `*_lin=True` enables a conservative dominant regular
pixel-spacing heuristic (`spacing_only`), requiring at least four supporting
ticks and a 60% consensus. It does not calibrate numerical values. `*_log=True`
without arrays reports `needs_values` and preserves the major/minor pattern:
logarithmic spacing alone does not uniquely determine arbitrary tick values.
With no arrays or scale flags, the original automatic detection is unchanged.

The same checks can be applied to an existing result through the public
`check_tick_consistency` function, without reading the image again. Previously
separated evidence is reconsidered when new reference values are supplied.
The paper example now retains 17 x ticks and 5 y ticks and separates its two
data-point branches that were mistaken for ticks.

## Rectangle model

`Rectangle(left, top, right, bottom)` describes four edge centerlines in pixels.
The origin is the top-left pixel center; x increases to the right and y downward.
Coordinates may be fractional. These are image coordinates, independent of
data values and any linear or logarithmic axis scales.

```python
from figure_data_unwrapper import Rectangle, rectangle_model

rectangle = Rectangle(left=60, top=40, right=890, bottom=530)
outline = rectangle_model((600, 900), rectangle, line_width=2)
```

`rectangle_model` returns a boolean image of the outline. True marks the border;
the interior stays empty. Bounds refer to geometric centerlines, not array slices.

## Binarization and frame extraction

```python
from figure_data_unwrapper import check_frame_colors, detect_frame, fit_frame, foreground_mask

mask = foreground_mask(image, min_contrast=0.03)
result = fit_frame(mask)
result = check_frame_colors(image, result)
if result is not None:
    print(result.rectangle)
    print(result.visible_sides)   # supported by finite line segments
    print(result.inferred_sides)  # completed from their endpoints

rectangle = detect_frame(image)  # convenience API: Rectangle or None
```

Input images are normalized to `[0, 1]`: grayscale, RGB, or RGBA. RGBA is
composited over white. Boolean masks already use True for foreground.
The median image perimeter estimates the uniform page background. Pixels whose
absolute contrast exceeds the configured threshold become foreground. RGB uses
the maximum channel difference so that bright saturated colors retain their
contrast before binarization. This also supports light strokes on dark backgrounds.

The geometric stage:

1. Finds long horizontal and vertical foreground segments, grouping stroke width
   and permitting small internal gaps.
2. Proposes rectangles from two adjacent segments whose endpoints meet, or from
   two opposite segments with matching endpoints.
3. Refines supported sides to measured stroke centerlines. A side matches only
   if its line position and full longitudinal extent agree with the rectangle.
4. Chooses the rectangle with the most supported sides, then largest area.
   At least two full sides must be present; other sides are inferred from
   the segment endpoints. Short corner continuations from ticks are tolerated,
   and intersecting side centerlines take precedence over those continuations.

Defaults require each dimension to span at least 25% of the image dimension
and each retained segment to have at least 90% coverage. At most 20 candidates
per orientation bound the search. Segments must be at least five times longer
than their stroke width to avoid treating filled regions as lines.

For an experiment with the old preprocessing, use
`detect_frame(image, darkness_threshold=0.5)`. This reproduces the former
dark-pixel mask while using the improved geometric fitter.

The model assumes full-length, axis-aligned spines on a mostly uniform page,
with background margins visible at the image perimeter. Larger competing
rectangular objects, multiple panels, cropped or shortened axes, perspective,
and nonuniform lighting need further work. A rectangle inferred from shortened
spines need not describe the original plot extent. Grid lines can also supply
geometric evidence; detected sides describe line support, not semantic labels.
A frame alone does not calibrate data coordinates or their origin.

## Original-color frame consistency

After binary geometry proposes a frame, `check_frame_colors(image, result)`
walks each straight side in the original RGB image. Every side has its own
dominant color: a black left side, white right side and colored horizontal
sides are allowed. Corner pixels are excluded from these independent paths.
RGB/RGBA and grayscale are supported; reported colors are normalized to 0–1.

By default, channel differences up to `color_tolerance=0.1` are tolerated.
Short deviations of up to `max(3, ceil(1% of the path length))` pixels are allowed;
`max_interruption_px` can override this. An internal A–B–A deviation may also
represent an occlusion if it occupies at most 20% of the path and at least 75%
of the path retains the dominant color. Longer endpoint changes or persistent
internal changes on a supported stroke are suspicious. These rules are
heuristics for antialiasing, crossings and occlusions, not semantic proof.

Uniform color alone does not establish a line. Transverse RGB probes must also
support a narrow stroke over at least 75% of the side. This distinguishes smooth
shading or uniform background from a real stroke and supports white strokes
against a darker background. A white side on white background remains inferred
because the pixels cannot establish that it exists.

Endpoint color transitions nominate nearby perpendicular frame boundaries for
correction. The proposed boundary must itself have a consistent full-length
stroke and improve the complete frame without invalidating good sides.
Corrections stay within `max(4 pixels, search_fraction * initial span)`, with
`search_fraction=0.1` by default. Missing sides alone do not trigger a search,
which prevents arbitrary interior grid lines from being promoted to frame sides.
Set `correct=False` to inspect the candidate without changing its coordinates.

`frame.color_check` records the original rectangle, original and final per-side
colors, color-change intervals with allowed/rejected reasons, jump positions,
stroke support and notes. Its status is `valid`, `partial`, `suspect` or
`rejected`. Partial frames retain inferred sides; unresolved suspicious changes
prevent subsequent tick/grid extraction. `fit_frame` remains the binary-only
stage; `pre_analyzer`, `detect_frame` and automatic `detect_ticks_and_grid` use
the RGB check. Check plots report its status and any boundary correction.

For `examples/real_example2.png`, the gray upper page shadow originally extended
the proposed frame to row 0. RGB transitions on the vertical sides identify the
actual upper spine around row 27. Correction removes the shadow band from the
plot interior, so it no longer appears as a spurious grid line.

## Ticks and grid lines

```python
from figure_data_unwrapper import detect_ticks_and_grid

result = detect_ticks_and_grid(image)
if result is not None:
    print([tick.position for tick in result.x_ticks])
    print([tick.position for tick in result.y_ticks])
    print(result.vertical_grid, result.horizontal_grid)

# Optional hints, e.g. to select longer major ticks among shorter minor ticks:
guided = detect_ticks_and_grid(image, expected_x_ticks=7, expected_y_ticks=5)
```

The function reconstructs and color-checks a frame first and returns None if
that fails or leaves unresolved color inconsistencies.
To reuse an existing successful reconstruction, pass a FrameDetection from
`fit_frame` as `frame=frame_result`. Its bounds must belong to the input image.
Detection searches along the four frame sides for perpendicular branches.
It measures and excludes the frame stroke width before scanning, including
inward/outward marks and marks at opposite sides. Corresponding marks within
1.5 pixels are merged into one AxisTick with per-side TickMark evidence.

Grid search walks the frame boundaries in pixel coordinates. Immediately beyond
the measured frame stroke, perpendicular branches seed candidate positions;
the same position must have support near the opposite frame boundary. Each
candidate is then traced straight across the plot. An internal gap longer than
the dash allowance rejects the line, even if its total foreground count is high.
Detached legend lines and text fragments therefore cannot seed a grid merely
by occupying much of an interior image row or column.

The dash allowance defaults to `max(4, ceil(1% of the interior span))` pixels and
applies to both boundary attachment and internal gaps. `grid_max_gap` overrides
it, including zero for uninterrupted traces. At least 30% foreground coverage
is additionally required by default. `GridLine.frame_sides` records the two
supporting boundaries, including inferred frame boundaries. Frame strokes are
excluded from the search. Grid evidence alone does not create observed ticks;
when inward ticks overlap a lighter grid, local contrast distinguishes the
stronger short marks. Long interruptions are rejected rather than silently
bridged through labels or disconnected lines.

Positions are pixel columns for x (left to right) and pixel rows for y (top to
bottom). No uniform spacing or linear/logarithmic scale is assumed. Major and
minor ticks are both included automatically; these terms are not inferred from
labels. The count hints select longer marks only if a clear length distinction
exists. Otherwise observed positions are retained and `notes` explains the
mismatch; `x_count_matches` and `y_count_matches` expose it. No positions are
invented to satisfy a count. Zero is allowed as an absence hint.

```sh
pixi run detect-ticks
pixi run detect-ticks --x-ticks 7 --y-ticks 5
pixi run detect-ticks path/to/figure.png --output examples/custom_ticks.png
```

The example writes an overlay and a JSON record, by default
`examples/parabola_ticks.png` and `examples/parabola_ticks.json`. The original
parabola yields seven x ticks, five y ticks, seven vertical grid lines, and
five horizontal grid lines.

This stage identifies image geometry, not tick labels or numeric axis values.
Very short marks below `min_tick_length=2` foreground pixels, dense overlapping
marks, and heavily occluded lines need further work. A same-color inward tick
superimposed on an identical grid line, or on a corner spine, is not separately
identifiable from pixels. Likewise an arbitrary full-span data line can resemble
a grid line; grid results describe geometric candidates rather than semantic
certainty. `grid_min_coverage` and `min_contrast` can be adjusted for the input.

## Color-frequency ranking

```python
from figure_data_unwrapper import rank_colors

ranking = rank_colors(image, n_colors=5)
# [(R, G, B, K_pixel), ...], ordered by decreasing pixel count
```

The function counts exact RGB values across the whole image and returns up to
`n_colors` tuples. Ties are sorted by ascending RGB. Zero returns an empty list;
a limit larger than the number of distinct colors returns all of them.
Integer input channels use 0–255, normalized floating input channels use 0–1,
and the returned RGB values retain that scale. Pixel counts are Python integers.
Grayscale is expanded to equal RGB channels. RGBA is composited over white,
as in the extraction pipeline; integer RGBA is rounded to 8-bit RGB afterward.

Float colors are not quantized or clustered: even nearby antialiasing shades
remain separate entries. This is supplementary image information; the ranking
does not automatically assign colors to frames, grid lines, or data points.

```sh
pixi run rank-colors --n-colors 5
pixi run rank-colors path/to/figure.png --n-colors 20
```

The example prints the tuples directly. Matplotlib loads PNGs as normalized
floating arrays, so their RGB values are printed on the 0–1 scale.

## Black-and-white consistency trial

`pixi run binarization-trial` generates yellow left/bottom axes without a grid.
It compares a fixed grayscale threshold of 0.5 with RGB background-contrast
thresholds of 0.02, 0.05, 0.10, and 0.20. All variants use the same geometric
fitter, isolating the effect of preprocessing.

The fixed grayscale mask loses the yellow axes and yields no frame. All four
contrast masks yield identical coordinates in this example, with left/bottom
observed and top/right inferred. This is a threshold-stability diagnostic, not
proof that the extracted frame is correct on arbitrary images.

Inputs, masks, a comparison PNG, and per-threshold CSV results are saved under
`examples/binarization_trial/`. The comparison uses green for observed sides
and dashed magenta for inferred sides.

## Tests and challenge matrix

```sh
pixi run test
pixi run test-frame-variants
```

The 144-case matrix crosses nine colors with all 16 combinations of visible
sides. Its detector inputs remain grayscale, and its original independent
renderer ground truth and one-pixel tolerance are unchanged. Two or more complete
finite spines request reconstruction; zero or one side without a grid request
None. Arbitrary cropped or obscured axes need not be identifiable this way.

The initial detector passed 50/144 cases: five reconstructions and 45 correct
abstentions. The improved detector passes 144/144: all 99 reconstructions and
45 abstentions. Original measurements are retained in
`examples/frame_variants/baseline_results.csv`.

Additional tests use independently drawn partial frames without a grid, varying
positions, intensities, and stroke widths. They check measured/inferred side
metadata, threshold perturbations, mixed side colors, dark backgrounds,
transparency, small background variations, invalid inputs, and rejection of
filled regions, crossing lines, and inconsistent opposite endpoints.
The former missing-side rejection test now requires reconstruction from the
remaining intact endpoints, reflecting the extended API contract.

Tick/grid tests use direct pixel fixtures and independently rendered Matplotlib
plots with a one-pixel position tolerance. They cover inward/outward/bidirectional
ticks, opposing sides, missing frame sides, corner ticks, several grid styles,
grids without ticks, colors, dark backgrounds, logarithmic spacing, optional
count hints, detached labels, and failure of the prerequisite frame extraction.

Boundary-tracing regressions cover the real paper figure in
`examples/real_example1.png` (no grid), detached legend/text fragments, short
boundary stubs separated from interior lines, one-sided attachment, dash-gap
limits, thick frame strokes, and dark backgrounds. Genuine grids are retained
in these tests; disabling grid detection cannot satisfy the regressions.

The challenge runner saves `results.csv`, `test_output.txt`, `overview.png`,
and `samples.png` under `examples/frame_variants/` and returns a nonzero exit
code on failure. No skipped or expected-failure tests hide failures.

PNG-export tests check native pixel preservation, overlay alignment, solid and
dashed frame sides, optional tick markers, color swatches, failed analysis,
transparency, grayscale input, and rejection of inconsistent geometry.

## Layout

- `src/figure_data_unwrapper/`: geometry, preprocessing, and frame extraction.
- `tests/`: unittest checks and rendered color/visibility challenges.
- `examples/`: reproducible figures, extraction overlays, and binarization trial.
- `pixi.toml` and `pixi.lock`: environment and resolved dependencies.
- `pyproject.toml`: Python package metadata and build configuration.
