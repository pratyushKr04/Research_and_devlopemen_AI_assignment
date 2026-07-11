"""
Plot the fitted curve against the given data points, as a visual sanity
check on top of the numerical residuals reported by fit_parameters.py.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fit_parameters import fit, Y_OFFSET, OMEGA, T_BOUNDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="xy_data.csv")
    parser.add_argument("--out", default="fit_comparison.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    x, y = df["x"].values, df["y"].values

    params = fit(x, y)
    theta, M, X = params["theta_rad"], params["M"], params["X"]

    t = np.linspace(T_BOUNDS[0], T_BOUNDS[1],60)
    v = np.exp(M * np.abs(t)) * np.sin(OMEGA * t)
    curve_x = t * np.cos(theta) - v * np.sin(theta) + X
    curve_y = Y_OFFSET + t * np.sin(theta) + v * np.cos(theta)

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=6, alpha=0.4, label="Given data points", color="tab:blue")
    plt.plot(curve_x, curve_y, color="tab:red", linewidth=1.5,
              label=f"Fitted curve (theta={params['theta_deg']:.8f} deg, "
                    f"M={M:.8f}, X={X:.8f})")
    plt.axis("equal")
    plt.legend()
    plt.title("Fitted parametric curve vs. given data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
