import numpy as np

from moirestrain import analyze


def main() -> None:
    height, width = 80, 120
    period = 8
    y, x = np.mgrid[:height, :width]

    reference = 128.0 + 80.0 * np.cos(2.0 * np.pi * x / period)
    true_u = 0.6 + 0.002 * x
    deformed = 128.0 + 80.0 * np.cos(2.0 * np.pi * (x + true_u) / period)

    result = analyze(reference, deformed, period=period, axis="x", unwrap_axis="x")
    print(f"mean displacement: {np.mean(result.displacement):.3f} px")
    print(f"true mean:         {np.mean(true_u):.3f} px")


if __name__ == "__main__":
    main()
