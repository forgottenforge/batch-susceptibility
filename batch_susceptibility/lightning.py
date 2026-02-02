# Copyright (c) 2025 ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE.txt for details.
# Commercial license available: nfo@forgottenforge.xyz
"""PyTorch Lightning integration for batch-size susceptibility.

Provides:
  - SusceptibilityCallback: Lightning callback for online monitoring
  - BatchSizeFinder: uses Lightning's Trainer for probing
"""

from typing import Optional, List
import numpy as np

from .core import BatchSusceptibility, SusceptibilityResult

try:
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise ImportError(
        "PyTorch Lightning is required for this module. "
        "Install with: pip install batch-susceptibility[lightning]"
    )


class SusceptibilityCallback(pl.Callback):
    """Lightning callback that monitors batch-size susceptibility.

    Example:
        >>> cb = SusceptibilityCallback(check_every=500)
        >>> trainer = pl.Trainer(callbacks=[cb])
        >>> trainer.fit(model, dataloader)
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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        elif hasattr(outputs, "loss"):
            loss = outputs.loss

        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu().item()
            self._losses.append(float(loss))
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

                # Log to Lightning's logger
                if trainer.logger:
                    trainer.logger.log_metrics({
                        "susceptibility/K_c": result.K_c,
                        "susceptibility/kappa": result.kappa,
                        "susceptibility/alpha": result.alpha,
                    }, step=trainer.global_step)

    def result(self) -> Optional[SusceptibilityResult]:
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
    """Find optimal batch size using Lightning's Trainer.

    Example:
        >>> finder = BatchSizeFinder(model_cls, dataset, model_kwargs={...})
        >>> result = finder.run()
        >>> print(result.K_c)
    """

    def __init__(
        self,
        model: "pl.LightningModule",
        train_dataset: "Dataset",
        batch_sizes: Optional[List[int]] = None,
        max_steps: int = 50,
        num_workers: int = 0,
        accelerator: str = "auto",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.max_steps = max_steps
        self.num_workers = num_workers
        self.accelerator = accelerator

        n = len(train_dataset)
        if batch_sizes is None:
            self.batch_sizes = [
                b for b in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                if b <= n
            ]
        else:
            self.batch_sizes = sorted(b for b in batch_sizes if b <= n)

    def run(self, verbose: bool = True) -> SusceptibilityResult:
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        metrics = []

        for bs in self.batch_sizes:
            self.model.load_state_dict(
                {k: v.clone() for k, v in initial_state.items()}
            )

            loader = DataLoader(
                self.train_dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )

            cb = SusceptibilityCallback(
                check_every=self.max_steps + 1,  # don't trigger during probe
                verbose=False,
            )

            trainer = pl.Trainer(
                max_steps=self.max_steps,
                accelerator=self.accelerator,
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
                callbacks=[cb],
            )

            trainer.fit(self.model, loader)

            if cb._losses:
                avg_loss = np.mean(cb._losses[-max(1, len(cb._losses) // 3):])
            else:
                avg_loss = float("nan")
            metrics.append(avg_loss)

            if verbose:
                print(f"  BS={bs:<6d}  loss={avg_loss:.4f}")

        self.model.load_state_dict(initial_state)

        analyzer = BatchSusceptibility()
        result = analyzer.find_critical_from_sweep(
            np.array(self.batch_sizes, dtype=float),
            np.array(metrics, dtype=float),
        )

        if verbose:
            print(f"\n  >> Optimal batch size: {result.K_c:.0f} (kappa={result.kappa:.2f})")

        return result
