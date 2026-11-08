"""Compare a fixed gray threshold with color-contrast masks on yellow L axes."""

import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import numpy as np

from figure_data_unwrapper import fit_frame, foreground_mask


def overlay_detection(axes, result) -> None:
    if result is None:
        return
    r = result.rectangle
    edges = {
        "left": ([r.left, r.left], [r.top, r.bottom]),
        "top": ([r.left, r.right], [r.top, r.top]),
        "right": ([r.right, r.right], [r.top, r.bottom]),
        "bottom": ([r.left, r.right], [r.bottom, r.bottom]),
    }
    for side, (x, y) in edges.items():
        measured = side in result.visible_sides
        axes.plot(x, y, color="green" if measured else "magenta",
                  linestyle="-" if measured else "--", linewidth=1)


def main() -> None:
    output_dir = Path(__file__).with_name("binarization_trial")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Figure(figsize=(4, 3), dpi=120)
    canvas = FigureCanvasAgg(source)
    axes = source.add_axes((0.18, 0.20, 0.74, 0.70))
    axes.scatter([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4], color="#333333", s=14)
    axes.set(xlim=(-2.5, 2.5), ylim=(-0.5, 4.5))
    for side, spine in axes.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color("yellow")
        spine.set_linewidth(1.2)
    canvas.draw()
    # No renderer geometry is passed to extraction; only this RGB raster.
    rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy() / 255
    mpimg.imsave(output_dir / "original.png", rgb)
    gray = rgb @ np.array([0.2126, 0.7152, 0.0722])
    legacy_mask = gray < 0.5

    figure = Figure(figsize=(12, 7))
    FigureCanvasAgg(figure)
    panels = list(figure.subplots(2, 3).flat)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.84,
                           wspace=0.10, hspace=0.28)
    panels[0].imshow(rgb)
    panels[0].set_title("Input: yellow left + bottom; no grid")
    panels[1].imshow(~legacy_mask, cmap="gray", vmin=0, vmax=1)
    legacy = fit_frame(legacy_mask)
    overlay_detection(panels[1], legacy)
    panels[1].set_title(f"Gray < 0.5: {'frame' if legacy else 'no frame'}")
    mpimg.imsave(output_dir / "fixed_threshold.png", ~legacy_mask, cmap="gray", vmin=0, vmax=1)

    rows = []
    rectangles = []
    for panel, threshold in zip(panels[2:], (0.02, 0.05, 0.10, 0.20)):
        mask = foreground_mask(rgb, min_contrast=threshold)
        result = fit_frame(mask)
        panel.imshow(~mask, cmap="gray", vmin=0, vmax=1)
        overlay_detection(panel, result)
        panel.set_title(f"Color contrast >= {threshold:.2f}: {'frame' if result else 'no frame'}")
        mpimg.imsave(output_dir / f"contrast_{threshold:.2f}.png", ~mask,
                     cmap="gray", vmin=0, vmax=1)
        r = result.rectangle if result else None
        if r:
            rectangles.append([r.left, r.top, r.right, r.bottom])
        rows.append({
            "min_contrast": threshold,
            "detected": result is not None,
            "left": r.left if r else "", "top": r.top if r else "",
            "right": r.right if r else "", "bottom": r.bottom if r else "",
            "visible_sides": "+".join(result.visible_sides) if result else "",
            "inferred_sides": "+".join(result.inferred_sides) if result else "",
        })
    spread = float(np.ptp(np.array(rectangles), axis=0).max()) if rectangles else None
    stable = len(rectangles) == 4 and spread <= 1
    for panel in panels:
        panel.set_axis_off()
    figure.suptitle(
        f"Binarization trial: stable across four contrast thresholds = {stable}\n"
        "Green: measured sides; magenta dashed: inferred sides"
    )
    figure.savefig(output_dir / "comparison.png", dpi=120)
    with (output_dir / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Fixed gray threshold: {'frame' if legacy else 'no frame'}")
    print(f"Contrast trials: {len(rectangles)}/4 frames; max coordinate spread = {spread} px")
    print(f"Stable within 1 pixel: {stable}; this is a stability check, not proof of correctness.")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
