"""Recover theta, M, X for the curve in xy_data.csv.  please see README.md for approach."""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

Y_BASE = 42.0
FREQ = 0.3

THETA_RANGE_DEG = (0.0, 50.0)
M_RANGE = (-0.05, 0.05)
X_RANGE = (0.0, 100.0)


def residuals(params, x, y):
    """Per-point residual for candidate (theta_rad, M, X)."""
    theta, M, X = params
    xt = x - X
    yt = y - Y_BASE

    t = xt * np.cos(theta) + yt * np.sin(theta)
    v = -xt * np.sin(theta) + yt * np.cos(theta)

    predicted_v = np.exp(M * np.abs(t)) * np.sin(FREQ * t)
    return v - predicted_v


def sse(params, x, y):
    """Sum of squared residuals -- the objective for the global search."""
    r = residuals(params, x, y)
    return float(np.sum(r ** 2))


def estimate_parameters(x, y, seed=42):
    """Run global search (Differential Evolution) then local least squares."""
    bounds = [
        (np.deg2rad(THETA_RANGE_DEG[0]), np.deg2rad(THETA_RANGE_DEG[1])),
        M_RANGE,
        X_RANGE,
    ]

    de_result = differential_evolution(
        sse, bounds, args=(x, y),
        seed=seed, tol=1e-12, maxiter=2000, popsize=40, polish=True,
    )

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    ls_result = least_squares(
        residuals, de_result.x, args=(x, y),
        bounds=(lo, hi), xtol=1e-15, ftol=1e-15, gtol=1e-15,
    )

    theta, M, X = ls_result.x
    return {
        "theta_rad": theta,
        "theta_deg": np.rad2deg(theta),
        "M": M,
        "X": X,
        "sse": float(np.sum(ls_result.fun ** 2)),
        "max_abs_residual": float(np.max(np.abs(ls_result.fun))),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit theta, M, X to xy_data.csv")
    parser.add_argument("--csv", default="xy_data.csv", help="Path to xy_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    x, y = df["x"].values, df["y"].values
    print(f"Loaded {len(x)} points from {args.csv}")

    params = estimate_parameters(x, y)
    print("\n=== Fitted parameters ===")
    print(f"theta = {params['theta_deg']:.6f} deg  ({params['theta_rad']:.10f} rad)")
    print(f"M     = {params['M']:.6f}")
    print(f"X     = {params['X']:.6f}")
    print(f"sum of squared residuals = {params['sse']:.3e}")
    print(f"max abs residual         = {params['max_abs_residual']:.3e}")



if __name__ == "__main__":
    main()
