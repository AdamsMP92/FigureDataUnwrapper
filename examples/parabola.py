"""Generate a reference figure with known samples of y = x**2."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    x = np.linspace(-3.0, 3.0, 13)
    y = x**2

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.scatter(x, y, s=40)
    ax.set(xlabel="$x$", ylabel="$y$", title=r"Parabola: $y = x^2$")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    output_path = Path(__file__).with_suffix(".png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
