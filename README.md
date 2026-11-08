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
evenly spaced between -3 and 3, saves `examples/parabola/parabola.png`, and runs the
`pre_analyzer` wrapper from `examples/parabola/parabola_check.py`. Its standard plausibility
plot is saved as `examples/parabola/parabola_check.png`, without opening a window.
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

image, frame, ticks, colors = pre_analyzer(Path("examples/parabola/parabola.png"), n_colors=7)
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
    "examples/real_example_1/real_example1.png", n_colors=7,
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

For `examples/real_example_2/real_example2.png`, the gray upper page shadow originally extended
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

## Removing approximate colors

`remove_colors` replaces selected color families by the estimated background
in a copy of the source, and retains a boolean mask of the discarded pixels:

```python
from figure_data_unwrapper import remove_colors, plot_color_removal

removal = remove_colors(image, ["red", "blue"], hue_tolerance=18)
cleaned_image, removed_mask = removal
plot_color_removal(image, removal, output="color_removal.png")
```

The default `mode="hue"` compares circular HSV hue, so dark/light shades and
antialiasing mixtures can match without requiring an exact RGB triple.
`hue_tolerance=15` is in degrees; `min_saturation=0.1` protects neutral black,
gray and white pixels. Lower saturation thresholds include paler color fringes.
Background-near pixels are additionally protected by `min_contrast=0.03`.
For achromatic targets, choose `mode="rgb"` and set `rgb_tolerance` (Euclidean
distance in normalized RGB). Targets accept names, hex strings, integer RGB
`[0,255]`, floating RGB `[0,1]`, or sequences of these.

The median image perimeter estimates a uniform page background;
`background_color` overrides it. Integer, float, grayscale and RGBA inputs are
supported. RGBA is composited over white, and output is always normalized RGB.
The original array is untouched. `mask=True` marks removed/unknown content;
`removed_pixels` counts the union, while `counts` reports each target separately.
Removing an overlay does not reconstruct a covered curve segment.

```sh
pixi run remove-real-example2-colors
```

`examples/real_example_2/real_example2_remove_colors.py` writes the cleaned PNG,
`real_example2_removed_mask.png` (white = removed), and a three-panel comparison
`real_example2_color_removal.png`. With the current settings it removes 390,530
pixels, including red/blue content in the insets. All exactly neutral grayscale
pixels retain their original values. Keep frame/tick calibration from the
original analysis when subsequently extracting from the cleaned copy.

## Color-selected calibrated pixel clouds

`extract_color_data` directly accepts the four-part return value from
`pre_analyzer`. Both axes must have a unique numerical tick mapping, obtained
by supplying axis tick values. A frame or regular tick spacing alone does not
establish numerical data coordinates; ambiguous mappings fail explicitly.

```python
from figure_data_unwrapper import extract_color_data, plot_color_data

# analysis = pre_analyzer(..., x_ticks=..., y_ticks=..., ...)
cloud = extract_color_data(analysis, "red", components="largest")
x, y, eps = cloud
plot_color_data(analysis, cloud, output="red_curve.png")
```

The result contains **one point per matching source pixel**, with no centerline
fit, interpolation, thinning or curve smoothing. `x` and `y` are 1D arrays of
axis values; `eps` has shape `(N, 2)`, ordered `(eps_x, eps_y)` in the respective
axis units. `cloud.pixels` preserves the source `(column, row)` positions.
`cloud.to_array()` returns columns `x,y,eps_x,eps_y,pixel_x,pixel_y`, suitable for
CSV export. Points stay in source row-major order, rather than curve order.

Color selection uses Euclidean distance in normalized RGB with
`color_tolerance=0.2`. Use a color name/hex string, integer RGB in `[0,255]`,
or floating RGB in `[0,1]`; nearby antialiasing shades can match within that
tolerance. RGBA is composited over white. The frame interior is cropped by
`frame_margin=4` pixels on every side, excluding the frame strip. This margin
can be adjusted for thick or cropped frames. No match produces an empty cloud
with the same array shapes, rather than inventing points.

The default `components="all"` keeps disconnected markers and fragments.
`components="largest"` selects the largest 8-connected color component by pixel
area. This removes the separate red legend stroke in example1. It is a geometric
heuristic: fragmented curves may lose valid components, and a larger same-color
object or a touching legend cannot be distinguished by color alone.
`matched_pixel_count`, `component_count` and `notes` report the selection.
The diagnostic PNG overlays selected pixels in cyan beside the calibrated cloud.

Calibration reuses the accepted tick assignments and the existing mappings
`pixel = a*u + b`, with `u=value` or `u=ln(value)` independently per axis.
The inverse handles linear, semilog, log-log and reversed axes; frame limits
are not assumed to coincide with the outermost reference values.
`calibrate_axes(ticks)` exposes the same calibration independently.

`eps` describes approximate **one-standard-deviation calibration uncertainty**,
conditional on the correct tick-value assignment and exact reference values.
Tick-localization errors are assumed independent with equal variance. Their
standard deviation is estimated from `sqrt(SSE/(n-2))`, with a lower bound of
`tick_sigma_px=1/sqrt(12)` by default (uniform tick location within +/-0.5 px).
Two anchors use that bound and report the lack of an independent residual test.
With `Suu = sum((u_i - mean(u_i))**2)`, covariance propagation gives

```text
eps_u = sigma_pixel / abs(a) * sqrt(1/n + (u - mean(u_i))**2 / Suu)
eps_value = eps_u                  # linear
eps_value = value * eps_u          # logarithmic, first-order approximation
```

This is the uncertainty of the calibrated coordinate for a fixed query pixel.
It excludes rendered line width, query-pixel quantization, experimental error
bars and systematic errors from an incorrect model or label assignment.
Calibration errors are shared across cloud points; the reported marginal
uncertainties must not be treated as independent errors for every colored pixel.

Run the complete red-curve example with

```sh
pixi run extract-real-example1-red
```

`examples/real_example_1/real_example1_red.py` reuses the existing paper pre-analysis
and writes `real_example1_red.csv` and `real_example1_red.png` in the same folder.
The current extraction keeps
10,192 of 11,163 red pixels inside the frame margin, excluding the separate
971-pixel legend stroke. Its approximate calibration uncertainties range from
0.053 to 0.101 x-axis units and 0.0041 to 0.0065 y-axis units.

For known annotations, `exclude_regions` accepts source-pixel `Rectangle`
objects. `max_channel_spread` optionally restricts `max(R,G,B)-min(R,G,B)`;
this helps select neutral gray antialiasing shades of black curves while
rejecting dark colored pixels. Applied exclusions are retained in the cloud.

## Two black curves in paper example2

```sh
pixi run extract-real-example2-black
```

`examples/real_example_2/real_example2_black_curves.py` first removes red/blue
content, then selects neutral dark pixels using `color_tolerance=1.0` and
`max_channel_spread=0.08`. The broad RGB tolerance includes the many gray shades
of the rasterized lines. A 35-pixel frame margin excludes inward ticks, and two
explicit rectangles exclude the insets; neither exclusion is inferred.

`split_two_curves` partitions this cloud by vertical order. Columns with two
distinct bands anchor a dividing boundary. That boundary is interpolated across
missing anchors, bounded by `max_anchor_gap_px`; curve values are never
interpolated. The current example requires a 40-pixel band separation and permits
at most an 80-pixel anchor gap. Extra bands, insufficient anchors or unsupported
gaps raise an error rather than forcing a two-curve assignment. The method
assumes two noncrossing, narrow bands in each image column; crossing curves need
a different association model.

The outputs `real_example2_black_upper.csv` and `real_example2_black_lower.csv`
contain 4,039 and 4,000 source pixels, respectively, each with columns
`x,y,eps_x,eps_y,pixel_x,pixel_y`. They reuse the original log-log calibration;
the lower series retains its values as plotted (no correction of the caption's
factor 0.5 is inferred). `eps` remains calibration uncertainty only.
`real_example2_black_curves.png`, created by `plot_color_curves`, shows both
clouds in distinct display colors and outlines the explicit inset exclusions.
Gaps caused by overdraw or selection remain visible.

## Reference-marker extraction

`extract_markers` complements the continuous color-pixel extractor: it returns
one calibrated center per **marker hypothesis**, using a cropped reference PNG.
It accepts the same four-part pre-analysis and reuses its unique numerical tick
mappings. The reference should contain one clean marker with background margins.

The minimal synthetic example is `examples/parabola/parabola_markers.py`.
It reuses the calibrated `parabola_check.py` analysis and `blue_marker.png`, then
saves `parabola_markers.csv` and `parabola_markers.png` alongside the input:

```sh
pixi run extract-parabola-markers
```

```python
from pathlib import Path
from figure_data_unwrapper import extract_markers, plot_marker_data

cloud = extract_markers(
    analysis, Path("examples/real_example_1/black_marker.png"),
    scales=(1.0, 1.1, 1.2),
)
x, y, eps = cloud
plot_marker_data(analysis, cloud, output="marker_check.png")
```

The method combines color screening, masked normalized correlation and a
weighted squared-error fit. Normalized correlation follows the template-matching
principle described in the [OpenCV documentation](https://docs.opencv.org/4.13.0/de/da9/tutorial_template_matching.html);
the implementation here uses NumPy and introduces no OpenCV dependency.

1. Estimate the background from the reference perimeter and the marker color
   from its contrasting foreground (`color` can override it).
2. Project RGB onto the background-to-marker color segment to measure ink
   coverage. Foreign colors farther than `color_tolerance=0.2` from that segment
   are masked as unknown. Thus a red line over a black circle does not count as
   evidence that the circle is missing at those pixels.
3. Screen positions and scales by normalized correlation (`min_score=0.65`),
   foreground support (`min_color_support=0.65`) and visible template foreground
   (`min_visible_fraction=0.6`). Search center positions inside the frame; a
   marker patch may cross the frame boundary if it remains within the image.
4. Fit the union of proposed templates, with `model = 1 - product(1-template)`
   for overlapping ink. Greedily add a hypothesis only if it reduces weighted
   image squared error by more than `min_gain * template_energy` (`min_gain=0.12`).
   Bounded joint position/scale refinement reduces redundant or displaced fits.

This allows touching markers to remain separate hypotheses. It does not require
each marker to be a separate connected color component. FFT correlations avoid
allocating all sliding windows, and `max_candidates=500`, at most 12 scales,
and `refinement_steps=2` bound the search. Parameter changes trade recall against
false positives; a larger template bank is not a guarantee of greater accuracy.

The default center is the reference ink centroid, so asymmetric crop margins do
not offset the coordinates. `anchor=(pixel_x, pixel_y)` overrides this location
inside the original reference PNG, useful for asymmetric marker shapes. The
default minimum center spacing is 45% of the smallest scaled marker diameter;
`min_distance` overrides it in source pixels. `frame_margin=1` applies to centers.

`cloud.matches` retains source centers, selected scale, correlation score, color
support, visible fraction, local patch MSE, unique error reduction and an overlap
flag. MSE includes neighboring strokes within the patch; the joint fit handles
overlap separately. Scores are diagnostic measurements, not probabilities.
`cloud.pixels` stores center positions, and `cloud.to_array()` exports columns
`x,y,eps_x,eps_y,pixel_x,pixel_y,scale,score,support,visible,mse,gain,overlap`.
The points are sorted by source x, then source y.

As with color clouds, `eps` contains axis-wise 1-sigma **calibration** uncertainty.
It does not estimate marker-center localization error or experimental error bars.
Merged, heavily occluded or poorly resolved symbols may remain ambiguous even
when a template score is high. The numbered source overlay is part of the check.

Identical legend symbols cannot be classified as annotations by shape and color
alone. `exclude_regions` accepts explicitly supplied `Rectangle` objects in
source pixel coordinates. These regions apply to marker centers and appear as
dashed amber boxes in the plot. The example supplies one known region around the
legend dot; this exclusion is not inferred automatically.

```sh
pixi run extract-real-example1-markers
```

`examples/real_example_1/real_example1_markers.py` uses `black_marker.png` with
scales 1.0, 1.1 and 1.2, and writes `real_example1_markers.csv` and
`real_example1_markers.png`. The current settings yield 45 hypotheses, 42 of
which overlap another template footprint. This is not a verified count of the
unavailable paper dataset. Synthetic tests instead use independent known marker
positions to check localization, overlap, colored overdraw, scale variation,
noise, crop padding, color changes and rejection of thin strokes.

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
`examples/real_example_1/real_example1.png` (no grid), detached legend/text fragments, short
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

- `src/figure_data_unwrapper/`: geometry, preprocessing, frame/tick extraction,
  numerical calibration, color-selected pixel clouds and diagnostic plots.
- `tests/`: unittest checks and rendered color/visibility challenges.
- `examples/parabola/`: synthetic figure generator, input PNG and plausibility check.
- `examples/real_example_1/`: first paper figure, pre-analysis, red-curve extraction,
  diagnostic PNGs and CSV output.
- `examples/real_example_2/`: second paper figure and its pre-analysis/check plot.
- `pixi.toml` and `pixi.lock`: environment and resolved dependencies.
- `pyproject.toml`: Python package metadata and build configuration.
