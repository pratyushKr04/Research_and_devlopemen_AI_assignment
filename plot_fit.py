"""Plot the fitted curve against xy_data.csv as a visual check on the fit."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fit_parameters import estimate_parameters, Y_BASE, FREQ

T_START, T_END = 6.0, 60.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="xy_data.csv")
    parser.add_argument("--out", default="fit_comparison.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    x, y = df["x"].values, df["y"].values

    params = estimate_parameters(x, y)
    theta, M, X = params["theta_rad"], params["M"], params["X"]

    t = np.linspace(T_START, T_END, 1500)
    v = np.exp(M * np.abs(t)) * np.sin(FREQ * t)
    curve_x = t * np.cos(theta) - v * np.sin(theta) + X
    curve_y = Y_BASE + t * np.sin(theta) + v * np.cos(theta)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(x, y, s=14, marker="x", linewidths=0.8,
               color="#4C72B0", alpha=0.55, label="Data points")
    ax.plot(curve_x, curve_y, color="#DD8452", linewidth=2.2, label="Fitted curve")

    caption = (
        f"$\\theta$ = {params['theta_deg']:.8f}$^\\circ$   "
        f"M = {M:.8f}   X = {X:.8f}"
    )
    ax.text(0.02, 0.02, caption, transform=ax.transAxes, fontsize=10,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="#999999"))

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Fitted parametric curve vs. given data points")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
