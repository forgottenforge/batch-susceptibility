#!/usr/bin/env python3
# Copyright (c) 2025 ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Example: Monitor susceptibility DURING training.

The callback computes K_c periodically and logs it.
If K_c changes significantly, your training dynamics are shifting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from batch_susceptibility.pytorch import SusceptibilityCallback


def main():
    # Synthetic data
    X = torch.randn(10000, 20)
    y = (X[:, 0] + X[:, 1] > 0).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Setup monitor
    monitor = SusceptibilityCallback(check_every=200, window_size=500)

    # Training loop
    model.train()
    for epoch in range(5):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()

            # Feed loss to monitor
            result = monitor.on_step(loss.item())
            if result:
                print(f"    -> K_c={result.K_c:.0f}, regime={result.regime}")

    # Final result
    final = monitor.result()
    if final:
        print(f"\nFinal: K_c={final.K_c:.0f}, kappa={final.kappa:.2f}")
        print(final.summary())


if __name__ == "__main__":
    main()
