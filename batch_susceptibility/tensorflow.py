# Copyright (c) 2025 ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
# Commercial license available: nfo@forgottenforge.xyz
"""TensorFlow / Keras integration for batch-size susceptibility.

Provides:
  - SusceptibilityCallback: Keras callback for online monitoring
  - BatchSizeFinder: automatic sweep over batch sizes
  - find_optimal_batch_size(): one-call convenience function
"""

from typing import Optional, List, Dict, Any
import numpy as np

from .core import BatchSusceptibility, SusceptibilityResult

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise ImportError(
        "TensorFlow is required for this module. "
        "Install with: pip install batch-susceptibility[tensorflow]"
    )


class SusceptibilityCallback(keras.callbacks.Callback):
    """Keras callback that monitors batch-size susceptibility during training.

    Example:
        >>> cb = SusceptibilityCallback(check_every=500)
        >>> model.fit(x, y, callbacks=[cb])
        >>> print(cb.result().K_c)
    """

    def __init__(
        self,
        check_every: int = 500,
        window_size: int = 2000,
        verbose: bool = True,
    ):
        super().__init__()
        self.check_every = check_every
        self.window_size = window_size
        self.verbose = verbose
        self._losses: List[float] = []
        self._results: List[SusceptibilityResult] = []
        self._step = 0

    def on_train_batch_end(self, batch, logs=None):
        if logs and "loss" in logs:
            self._losses.append(float(logs["loss"]))
            self._step += 1

            if self._step % self.check_every == 0 and len(self._losses) >= 50:
                window = self._losses[-self.window_size:]
                bs = BatchSusceptibility()
                bs.feed(window)
                result = bs.find_critical()
                self._results.append(result)

                if self.verbose:
                    print(
                        f"\n  [Step {self._step}] K_c={result.K_c:.0f}, "
                        f"kappa={result.kappa:.2f}, alpha={result.alpha:.3f}"
                    )

    def result(self) -> Optional[SusceptibilityResult]:
        """Get latest result or compute from all losses."""
        if self._results:
            return self._results[-1]
        if len(self._losses) >= 50:
            bs = BatchSusceptibility()
            bs.feed(self._losses)
            return bs.find_critical()
        return None

    @property
    def history(self) -> List[SusceptibilityResult]:
        return self._results


class BatchSizeFinder:
    """Find optimal batch size for a Keras model.

    Example:
        >>> finder = BatchSizeFinder(model, x_train, y_train)
        >>> result = finder.run()
        >>> print(f"Use batch size: {result.K_c}")
    """

    def __init__(
        self,
        model: "keras.Model",
        x,
        y,
        batch_sizes: Optional[List[int]] = None,
        steps_per_size: int = 50,
        validation_data=None,
    ):
        """
        Args:
            model: Compiled Keras model.
            x: Training inputs.
            y: Training targets.
            batch_sizes: Batch sizes to probe.
            steps_per_size: Steps per probe.
            validation_data: Optional (x_val, y_val) tuple.
        """
        self.model = model
        self.x = x
        self.y = y
        self.validation_data = validation_data
        self.steps_per_size = steps_per_size

        n_samples = len(x) if hasattr(x, "__len__") else x.shape[0]
        if batch_sizes is None:
            self.batch_sizes = [
                b for b in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                if b <= n_samples
            ]
        else:
            self.batch_sizes = sorted(b for b in batch_sizes if b <= n_samples)

    def run(self, verbose: bool = True) -> SusceptibilityResult:
        """Run sweep and find K_c."""
        initial_weights = self.model.get_weights()
        metrics = []

        for bs in self.batch_sizes:
            self.model.set_weights([w.copy() for w in initial_weights])

            hist = self.model.fit(
                self.x, self.y,
                batch_size=bs,
                epochs=1,
                steps_per_epoch=self.steps_per_size,
                verbose=0,
                validation_data=self.validation_data,
            )

            losses = hist.history["loss"]
            avg_loss = np.mean(losses[-max(1, len(losses) // 3):])
            metrics.append(avg_loss)

            if verbose:
                print(f"  BS={bs:<6d}  loss={avg_loss:.4f}")

        self.model.set_weights(initial_weights)

        analyzer = BatchSusceptibility()
        result = analyzer.find_critical_from_sweep(
            np.array(self.batch_sizes, dtype=float),
            np.array(metrics, dtype=float),
        )

        if verbose:
            print(f"\n  >> Optimal batch size: {result.K_c:.0f} (kappa={result.kappa:.2f})")

        return result


def find_optimal_batch_size(
    model: "keras.Model",
    x,
    y,
    batch_sizes: Optional[List[int]] = None,
    steps_per_size: int = 50,
    verbose: bool = True,
) -> SusceptibilityResult:
    """One-call function to find optimal batch size for a Keras model.

    Example:
        >>> result = find_optimal_batch_size(model, x_train, y_train)
        >>> print(result.K_c)
    """
    finder = BatchSizeFinder(
        model=model, x=x, y=y,
        batch_sizes=batch_sizes,
        steps_per_size=steps_per_size,
    )
    return finder.run(verbose=verbose)
