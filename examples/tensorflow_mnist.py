#!/usr/bin/env python3
# Copyright (c) 2025 ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE.txt for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Example: Find optimal batch size for MNIST with Keras."""

import numpy as np
from batch_susceptibility.tensorflow import find_optimal_batch_size, SusceptibilityCallback


def main():
    import tensorflow as tf
    from tensorflow import keras

    # 1. Data
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0

    # 2. Model
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(784,)),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # === Option A: Batch-size sweep ===
    print("=== Batch-Size Sweep ===")
    result = find_optimal_batch_size(
        model, x_train, y_train,
        batch_sizes=[16, 32, 64, 128, 256, 512, 1024],
        steps_per_size=100,
    )
    print(f"\nOptimal batch size: {result.K_c:.0f}")
    print(result.summary())

    # === Option B: Online monitoring ===
    print("\n=== Online Monitoring ===")
    # Recompile (fresh optimizer)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
    )
    cb = SusceptibilityCallback(check_every=200)
    model.fit(x_train, y_train, batch_size=64, epochs=3, callbacks=[cb])

    final = cb.result()
    if final:
        print(f"\nOnline K_c: {final.K_c:.0f}")


if __name__ == "__main__":
    main()
