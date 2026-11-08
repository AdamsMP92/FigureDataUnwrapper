# FigureDataUnwrapper

A Python package for reconstructing data from scientific figures when the
original datasets are unavailable. The current step detects a rectangular plot
frame in image pixel coordinates, including frames with missing sides.

## Getting started

With Pixi installed, run from the repository root:

```sh
pixi install
pixi run example
pixi run detect-frame
pixi run binarization-trial
pixi run test
```

The example plots 13 points on the parabola `y = x**2`, with `x` evenly spaced
between -3 and 3, and saves `examples/parabola.png` without opening a window.
`detect-frame` reads only the rendered PNG and writes `parabola_mask.png`
(black foreground on white background) and `parabola_frame.png` (red overlay).
Extraction has no access to renderer geometry or the original data.

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
from figure_data_unwrapper import detect_frame, fit_frame, foreground_mask

mask = foreground_mask(image, min_contrast=0.03)
result = fit_frame(mask)
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
4. Chooses the largest supported rectangle. At least two full sides must be
   present; other sides are inferred from the segment endpoints.

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

The challenge runner saves `results.csv`, `test_output.txt`, `overview.png`,
and `samples.png` under `examples/frame_variants/` and returns a nonzero exit
code on failure. No skipped or expected-failure tests hide failures.

## Layout

- `src/figure_data_unwrapper/`: geometry, preprocessing, and frame extraction.
- `tests/`: unittest checks and rendered color/visibility challenges.
- `examples/`: reproducible figures, extraction overlays, and binarization trial.
- `pixi.toml` and `pixi.lock`: environment and resolved dependencies.
- `pyproject.toml`: Python package metadata and build configuration.
