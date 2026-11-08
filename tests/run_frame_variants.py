"""Run the challenge matrix once and save its results and a visual overview."""

import csv
from pathlib import Path
import sys
import unittest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as RectanglePatch
import numpy as np

from test_frame_variants import (
    COLORS, SIDES, SIDE_COMBINATIONS, FrameCase, FrameVariantTests, render_case,
)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def _record(self, test, passed, message=""):
        frame = test.detected
        self.rows.append({
            "color": test.case.color,
            "visible_sides": "+".join(test.case.sides) or "none",
            "expected": "frame" if test.case.expects_frame else "none",
            "passed": passed,
            "detected": frame is not None,
            "left": frame.left if frame else "",
            "top": frame.top if frame else "",
            "right": frame.right if frame else "",
            "bottom": frame.bottom if frame else "",
            "max_error_px": test.max_error if test.max_error is not None else "",
            "message": message,
        })

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, True)

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self._record(test, False, str(error[1]))

    def addError(self, test, error):
        super().addError(test, error)
        if hasattr(test, "case"):
            self._record(test, False, f"ERROR: {error[1]}")


def main() -> int:
    output_dir = Path(__file__).resolve().parents[1] / "examples" / "frame_variants"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "test_output.txt"
    with log_path.open("w") as log:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FrameVariantTests)
        result = unittest.TextTestRunner(
            stream=log, verbosity=2, resultclass=RecordingResult
        ).run(suite)

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(result.rows[0]))
        writer.writeheader()
        writer.writerows(result.rows)

    lookup = {(row["color"], row["visible_sides"]): row for row in result.rows}
    matrix = np.zeros((len(COLORS), len(SIDE_COMBINATIONS), 3))
    figure = Figure(figsize=(15, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    for i, color in enumerate(COLORS):
        for j, sides in enumerate(SIDE_COMBINATIONS):
            row = lookup.get((color, "+".join(sides) or "none"))
            passed = row is not None and row["passed"]
            matrix[i, j] = [0.77, 0.90, 0.78] if passed else [0.97, 0.76, 0.73]
            label = "OK" if passed else "FAIL"
            if passed and row["expected"] == "none":
                label = "NONE"
            axes.text(j, i, label, ha="center", va="center", fontsize=7)
    axes.imshow(matrix, aspect="auto")
    axes.set_xticks(range(len(SIDE_COMBINATIONS)), [
        "+".join(sides) or "none" for sides in SIDE_COMBINATIONS
    ], rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(len(COLORS)), list(COLORS))
    passed = sum(row["passed"] for row in result.rows)
    axes.set_title(
        f"Frame extraction: {passed}/{result.testsRun} passed\n"
        "Green: correct bounds or correct abstention (NONE); red: failed reconstruction"
    )
    axes.set_xlabel("Visible sides (two or more: reconstruct; zero or one: abstain)")
    figure.savefig(output_dir / "overview.png", dpi=120)

    # Show actual inputs as well as the matrix; rerendering does not rerun
    # extraction. Overlay bounds only when the recorded test found a frame.
    samples = [FrameCase(color, SIDES) for color in COLORS] + [
        FrameCase("black", ("left", "bottom")),
        FrameCase("black", ("left", "bottom", "right")),
        FrameCase("black", ("left",)),
    ]
    sample_figure = Figure(figsize=(12, 9), layout="constrained")
    FigureCanvasAgg(sample_figure)
    for axes, case in zip(sample_figure.subplots(3, 4).flat, samples):
        rgb, _ = render_case(case)
        row = lookup[(case.color, "+".join(case.sides))]
        axes.imshow(rgb)
        if row["detected"]:
            axes.add_patch(RectanglePatch(
                (row["left"], row["top"]),
                row["right"] - row["left"], row["bottom"] - row["top"],
                fill=False, edgecolor="magenta", linewidth=1, linestyle="--",
            ))
        status = "PASS: reconstructed" if row["passed"] else "FAIL"
        if row["passed"] and row["expected"] == "none":
            status = "PASS: no frame"
        axes.set_title(f"{case.color}: {'+'.join(case.sides)}\n{status}", fontsize=8)
        axes.set_axis_off()
    sample_figure.suptitle("Rendered test inputs; detected bounds shown in magenta")
    sample_figure.savefig(output_dir / "samples.png", dpi=120)

    print(f"Frame variants: {passed}/{result.testsRun} passed; "
          f"{len(result.failures)} failures; {len(result.errors)} errors.")
    for color in COLORS:
        rows = [row for row in result.rows if row["color"] == color]
        reconstructions = [row for row in rows if row["expected"] == "frame"]
        abstentions = [row for row in rows if row["expected"] == "none"]
        print(f"  {color:12} reconstruction "
              f"{sum(row['passed'] for row in reconstructions)}/{len(reconstructions)}; "
              f"abstention {sum(row['passed'] for row in abstentions)}/{len(abstentions)}")
    print(f"CSV: {csv_path}")
    print(f"Details: {log_path}")
    print(f"Overview: {output_dir / 'overview.png'}")
    print(f"Samples: {output_dir / 'samples.png'}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
