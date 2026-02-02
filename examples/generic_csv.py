#!/usr/bin/env python3
# Copyright (c) 2025 ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE.txt for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Example: Analyze any time series from a CSV file.

Works with training logs, server metrics, sensor data, etc.
"""

import numpy as np
from batch_susceptibility import BatchSusceptibility


def main():
    # === From a numpy array (any data source) ===
    print("=== Synthetic example: correlated noise ===")

    # Generate correlated noise (AR(1) process)
    N = 5000
    rho = 0.8  # autocorrelation
    noise = np.random.randn(N)
    data = np.zeros(N)
    data[0] = noise[0]
    for i in range(1, N):
        data[i] = rho * data[i - 1] + np.sqrt(1 - rho ** 2) * noise[i]

    # Find critical scale
    bs = BatchSusceptibility(K_min=2, K_max=500)
    bs.feed(data)
    result = bs.find_critical()

    print(result.summary())
    print(f"\nInterpretation: K_c = {result.K_c:.0f}")
    print(f"  This means correlations extend ~{result.K_c:.0f} time steps.")
    print(f"  Batches smaller than {result.K_c:.0f} will have correlated samples.")

    # === From a CSV file ===
    print("\n=== From CSV ===")

    # Create sample CSV
    csv_path = "/tmp/sample_losses.csv"
    with open(csv_path, "w") as f:
        f.write("step,loss,accuracy\n")
        for i in range(3000):
            loss = 2.0 * np.exp(-i / 500) + 0.1 * np.random.randn()
            acc = 1.0 - np.exp(-i / 500) + 0.05 * np.random.randn()
            f.write(f"{i},{loss:.4f},{acc:.4f}\n")

    print(f"  Wrote sample CSV to {csv_path}")
    print(f"  Run: batch-susceptibility {csv_path} --column loss --verbose")
    print(f"  Or:  batch-susceptibility {csv_path} --column loss --plot result.png")


if __name__ == "__main__":
    main()
