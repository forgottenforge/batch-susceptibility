#!/usr/bin/env python3
# Copyright (c) 2025 ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Example: Find optimal batch size for CIFAR-10 with ResNet-18.

This is the simplest PyTorch usage: one function call.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from batch_susceptibility.pytorch import find_optimal_batch_size


def main():
    # 1. Setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    model = models.resnet18(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()

    # 2. Find optimal batch size (one line!)
    result = find_optimal_batch_size(
        model, dataset, loss_fn,
        optimizer_cls=torch.optim.Adam,
        batch_sizes=[16, 32, 64, 128, 256, 512, 1024],
        steps_per_size=100,
        lr=1e-3,
    )

    # 3. Use the result
    print(f"\nOptimal batch size: {result.K_c:.0f}")
    print(f"Confidence (kappa): {result.kappa:.2f}")
    print(f"Regime: {result.regime}")

    # 4. Optional: plot
    try:
        from batch_susceptibility.plot import plot_susceptibility
        fig = plot_susceptibility(result, title="CIFAR-10 / ResNet-18")
        fig.savefig("cifar10_susceptibility.png", dpi=150, bbox_inches="tight")
        print("Plot saved: cifar10_susceptibility.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
