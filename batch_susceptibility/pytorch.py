# Copyright (c) 2025 ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
# Commercial license available: nfo@forgottenforge.xyz
"""PyTorch integration for batch-size susceptibility.

Provides:
  - BatchSizeFinder: automatic optimal batch size search
  - SusceptibilityCallback: online monitoring during training
  - find_optimal_batch_size(): one-call convenience function
"""

from typing import Optional, Callable, List, Dict, Any
import numpy as np

from .core import BatchSusceptibility, SusceptibilityResult

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
except ImportError:
    raise ImportError(
        "PyTorch is required for this module. "
        "Install with: pip install batch-susceptibility[pytorch]"
    )


class BatchSizeFinder:
    """Find optimal batch size by running short training probes.

    Trains for a few steps at each candidate batch size and uses
    susceptibility analysis to find the critical scale.

    Example:
        >>> finder = BatchSizeFinder(model, dataset, loss_fn, optimizer_cls=torch.optim.Adam)
        >>> result = finder.run()
        >>> print(f"Use batch size: {result.K_c}")
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        dataset: "Dataset",
        loss_fn: "torch.nn.Module",
        optimizer_cls: type = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_sizes: Optional[List[int]] = None,
        steps_per_size: int = 50,
        device: Optional[str] = None,
        num_workers: int = 0,
    ):
        """
        Args:
            model: PyTorch model.
            dataset: Training dataset.
            loss_fn: Loss function (e.g. nn.CrossEntropyLoss()).
            optimizer_cls: Optimizer class (default: Adam).
            optimizer_kwargs: Kwargs for optimizer (default: {lr: 1e-3}).
            batch_sizes: List of batch sizes to test. Default: log-spaced 4..2048.
            steps_per_size: Training steps per batch size probe.
            device: Device string ('cpu', 'cuda', etc.). Auto-detected if None.
            num_workers: DataLoader workers.
        """
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls or torch.optim.Adam
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.steps_per_size = steps_per_size
        self.num_workers = num_workers

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if batch_sizes is None:
            self.batch_sizes = [
                4, 8, 16, 32, 48, 64, 96, 128, 192, 256,
                384, 512, 768, 1024, 1536, 2048,
            ]
        else:
            self.batch_sizes = sorted(batch_sizes)

        # Filter: batch size can't exceed dataset
        self.batch_sizes = [b for b in self.batch_sizes if b <= len(dataset)]

    def run(self, verbose: bool = True) -> SusceptibilityResult:
        """Run the batch-size sweep and find K_c.

        Args:
            verbose: Print progress.

        Returns:
            SusceptibilityResult with K_c = optimal batch size.
        """
        initial_state = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }

        metrics = []
        losses_per_size = {}

        for bs in self.batch_sizes:
            # Reset model
            self.model.load_state_dict(
                {k: v.clone() for k, v in initial_state.items()}
            )
            self.model.to(self.device)

            loader = DataLoader(
                self.dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )

            optimizer = self.optimizer_cls(
                self.model.parameters(), **self.optimizer_kwargs
            )

            step_losses = self._train_steps(loader, optimizer)
            losses_per_size[bs] = step_losses
            avg_loss = np.mean(step_losses[-max(1, len(step_losses) // 3):])
            metrics.append(avg_loss)

            if verbose:
                print(f"  BS={bs:<6d}  loss={avg_loss:.4f}  ({len(step_losses)} steps)")

        # Restore model
        self.model.load_state_dict(initial_state)

        # Find K_c
        bs = BatchSusceptibility()
        result = bs.find_critical_from_sweep(
            np.array(self.batch_sizes, dtype=float),
            np.array(metrics, dtype=float),
        )
        result.diagnostics["losses_per_size"] = losses_per_size

        if verbose:
            print(f"\n  >> Optimal batch size: {result.K_c:.0f} (kappa={result.kappa:.2f})")

        return result

    def _train_steps(
        self,
        loader: DataLoader,
        optimizer: "torch.optim.Optimizer",
    ) -> List[float]:
        """Train for self.steps_per_size steps, return per-step losses."""
        self.model.train()
        step_losses = []
        steps = 0
        data_iter = iter(loader)

        while steps < self.steps_per_size:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            elif isinstance(batch, dict):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
            else:
                raise ValueError(
                    f"Unsupported batch type: {type(batch)}. "
                    "Expected (input, target) tuple, list, or dict with 'input'/'target' keys."
                )

            optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.loss_fn(output, targets)
            loss.backward()
            optimizer.step()

            step_losses.append(loss.item())
            steps += 1

        return step_losses


class SusceptibilityCallback:
    """Monitor batch-size susceptibility during training.

    Collects per-step losses and periodically computes K_c.
    Designed to be called manually or integrated into a training loop.

    Example:
        >>> monitor = SusceptibilityCallback(check_every=500)
        >>> for step, loss in enumerate(training_loop):
        ...     monitor.on_step(loss)
        >>> result = monitor.result()
    """

    def __init__(
        self,
        check_every: int = 500,
        window_size: int = 2000,
        verbose: bool = True,
    ):
        """
        Args:
            check_every: Compute susceptibility every N steps.
            window_size: Use last N losses for analysis.
            verbose: Print updates.
        """
        self.check_every = check_every
        self.window_size = window_size
        self.verbose = verbose
        self._losses: List[float] = []
        self._results: List[SusceptibilityResult] = []
        self._step = 0

    def on_step(self, loss: float) -> Optional[SusceptibilityResult]:
        """Record a loss value. Returns result if a check was performed.

        Args:
            loss: Loss value for this step.

        Returns:
            SusceptibilityResult if a check was triggered, else None.
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        self._losses.append(loss)
        self._step += 1

        if self._step % self.check_every == 0 and len(self._losses) >= 50:
            window = self._losses[-self.window_size:]
            bs = BatchSusceptibility()
            bs.feed(window)
            result = bs.find_critical()
            self._results.append(result)

            if self.verbose:
                print(
                    f"  [Step {self._step}] K_c={result.K_c:.0f}, "
                    f"kappa={result.kappa:.2f}, alpha={result.alpha:.3f}"
                )
            return result
        return None

    def result(self) -> Optional[SusceptibilityResult]:
        """Get the latest result, or compute one from all collected losses."""
        if self._results:
            return self._results[-1]
        if len(self._losses) >= 50:
            bs = BatchSusceptibility()
            bs.feed(self._losses)
            return bs.find_critical()
        return None

    @property
    def history(self) -> List[SusceptibilityResult]:
        """All computed results."""
        return self._results


def find_optimal_batch_size(
    model: "torch.nn.Module",
    dataset: "Dataset",
    loss_fn: "torch.nn.Module",
    optimizer_cls: type = None,
    batch_sizes: Optional[List[int]] = None,
    steps_per_size: int = 50,
    verbose: bool = True,
    **optimizer_kwargs,
) -> SusceptibilityResult:
    """One-call convenience function to find optimal batch size.

    Example:
        >>> result = find_optimal_batch_size(model, train_dataset, nn.CrossEntropyLoss())
        >>> print(result.K_c)

    Returns:
        SusceptibilityResult with K_c = recommended batch size.
    """
    finder = BatchSizeFinder(
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs or {"lr": 1e-3},
        batch_sizes=batch_sizes,
        steps_per_size=steps_per_size,
    )
    return finder.run(verbose=verbose)
