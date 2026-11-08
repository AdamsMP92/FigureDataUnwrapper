"""Detect the frame using only the rendered parabola PNG and save an overlay."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from figure_data_unwrapper import fit_frame, foreground_mask, rectangle_model


def main() -> None:
    input_path = Path(__file__).with_name("parabola.png")
    image = plt.imread(input_path)
    # Composite any transparency over the assumed white background.
    rgb = image[..., :3]
    if image.shape[-1] == 4:
        alpha = image[..., 3:4]
        rgb = rgb * alpha + (1 - alpha)
    mask = foreground_mask(image)
    result = fit_frame(mask)
    if result is None:
        raise RuntimeError("No rectangular frame detected.")
    frame = result.rectangle

    overlay = rgb.copy()
    overlay[rectangle_model(mask.shape, frame, line_width=3)] = [1, 0, 0]
    output_path = input_path.with_name("parabola_frame.png")
    plt.imsave(output_path, overlay)
    plt.imsave(input_path.with_name("parabola_mask.png"), ~mask, cmap="gray", vmin=0, vmax=1)
    print(f"Detected frame (pixels): {frame}")
    print(f"Visible sides: {result.visible_sides}; inferred sides: {result.inferred_sides}")
    print(f"Saved overlay to {output_path}")


if __name__ == "__main__":
    main()
