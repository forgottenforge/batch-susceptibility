#!/usr/bin/env python3
# Copyright (c) 2025 ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE.txt for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Example: Find optimal batch size for HuggingFace Transformers fine-tuning.

Integrates with the Trainer API via a custom callback.
"""

from batch_susceptibility import BatchSusceptibility, SusceptibilityResult
from typing import Optional, List


# === HuggingFace Trainer Callback ===
try:
    from transformers import TrainerCallback

    class SusceptibilityTrainerCallback(TrainerCallback):
        """HuggingFace Transformers Trainer callback.

        Example:
            >>> from transformers import Trainer
            >>> cb = SusceptibilityTrainerCallback(check_every=200)
            >>> trainer = Trainer(model=model, args=args, callbacks=[cb], ...)
            >>> trainer.train()
            >>> print(cb.result().K_c)
        """

        def __init__(self, check_every: int = 500, window_size: int = 2000, verbose: bool = True):
            self.check_every = check_every
            self.window_size = window_size
            self.verbose = verbose
            self._losses: List[float] = []
            self._results: List[SusceptibilityResult] = []
            self._step = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self._losses.append(float(logs["loss"]))
                self._step += 1

                if self._step % self.check_every == 0 and len(self._losses) >= 50:
                    bs = BatchSusceptibility()
                    bs.feed(self._losses[-self.window_size:])
                    result = bs.find_critical()
                    self._results.append(result)

                    if self.verbose:
                        print(f"  [Step {state.global_step}] "
                              f"K_c={result.K_c:.0f}, kappa={result.kappa:.2f}")

        def result(self) -> Optional[SusceptibilityResult]:
            if self._results:
                return self._results[-1]
            if len(self._losses) >= 50:
                bs = BatchSusceptibility()
                bs.feed(self._losses)
                return bs.find_critical()
            return None

except ImportError:
    pass


def main():
    """Demonstrate usage with a mock training run."""
    import numpy as np

    print("=== HuggingFace Transformers Integration ===\n")
    print("Usage with real Trainer:")
    print("""
    from transformers import Trainer, TrainingArguments
    from batch_susceptibility.examples.huggingface_transformers import SusceptibilityTrainerCallback

    cb = SusceptibilityTrainerCallback(check_every=200)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(...),
        train_dataset=dataset,
        callbacks=[cb],
    )
    trainer.train()

    result = cb.result()
    print(f"Optimal batch size: {result.K_c}")
    """)

    # Mock demo
    print("--- Mock demo with synthetic losses ---\n")
    losses = 3.0 * np.exp(-np.arange(2000) / 300) + 0.2 * np.random.randn(2000)

    bs = BatchSusceptibility()
    bs.feed(losses)
    result = bs.find_critical()
    print(result.summary())


if __name__ == "__main__":
    main()
