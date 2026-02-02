# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE.txt for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Core susceptibility computation engine.

Two modes of operation:

1. ONLINE (single training run):
   Feed per-step losses from a training run with fixed batch size.
   The algorithm re-batches internally at multiple scales K and computes
   V(K) = Var(batch means). The critical scale K_c is where chi(K)
   = |dV/dK| peaks. Multiply by your actual mini-batch size to get
   the optimal batch size in samples.

2. SWEEP (multiple training runs):
   Train at several batch sizes, record final loss / metric at each.
   chi(B) = |dMetric/dB| peaks at B_c = optimal batch size.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any
import numpy as np
from scipy import stats


@dataclass
class SusceptibilityResult:
    """Result of a susceptibility analysis."""

    # Critical scale
    K_c: float
    """Critical batch size (in units of the input: steps for online, samples for sweep)."""

    # Confidence
    kappa: float
    """Peak sharpness: max(chi) / median(chi). Higher = more confident. >3 is significant."""

    alpha: float
    """Power-law exponent of V(K) ~ K^alpha. For i.i.d.: alpha = -1."""

    alpha_se: float
    """Standard error of alpha."""

    p_iid: float
    """p-value for H0: alpha = -1 (i.i.d. null hypothesis)."""

    # Raw data for plotting
    K_values: np.ndarray
    """Batch sizes tested."""

    V_values: np.ndarray
    """Variance of batch means at each K."""

    chi_values: np.ndarray
    """Susceptibility |dV/dK| at each K (in log-log space)."""

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """Whether a critical scale was reliably detected."""
        return self.kappa > 3.0 and self.p_iid < 0.05

    @property
    def regime(self) -> str:
        """Characterize the scaling regime."""
        if abs(self.alpha + 1) < max(3 * self.alpha_se, 0.1):
            return "iid"  # consistent with independent samples
        elif self.alpha > -0.5:
            return "correlated"  # strong positive correlations / drift
        elif self.alpha < -1.5:
            return "anti-correlated"  # sub-Poisson
        else:
            return "weakly-correlated"

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Critical batch size:  K_c = {self.K_c:.0f}",
            f"Sharpness:            kappa = {self.kappa:.2f}",
            f"Scaling exponent:     alpha = {self.alpha:.3f} +/- {self.alpha_se:.3f}",
            f"i.i.d. test:          p = {self.p_iid:.4f}",
            f"Regime:               {self.regime}",
            f"Significant:          {self.is_significant}",
        ]
        return "\n".join(lines)


class BatchSusceptibility:
    """Main interface for batch-size susceptibility analysis.

    Example (online mode):
        >>> bs = BatchSusceptibility()
        >>> bs.feed(per_step_losses)
        >>> result = bs.find_critical()
        >>> print(result.K_c)

    Example (sweep mode):
        >>> bs = BatchSusceptibility()
        >>> result = bs.find_critical_from_sweep(batch_sizes, metrics)
        >>> print(result.K_c)
    """

    def __init__(
        self,
        K_min: int = 2,
        K_max: Optional[int] = None,
        n_points: int = 60,
        min_batches: int = 10,
    ):
        """
        Args:
            K_min: Smallest batch size to test.
            K_max: Largest batch size to test. Default: N // min_batches.
            n_points: Number of K values to sample (log-spaced).
            min_batches: Minimum number of batches required at each K.
        """
        self.K_min = K_min
        self.K_max = K_max
        self.n_points = n_points
        self.min_batches = min_batches
        self._data: Optional[np.ndarray] = None

    def feed(self, values: Sequence[float]) -> "BatchSusceptibility":
        """Feed a sequence of per-step metric values.

        Args:
            values: 1D array of per-step losses, accuracies, gradients, etc.

        Returns:
            self (for chaining).
        """
        self._data = np.asarray(values, dtype=np.float64).ravel()
        if len(self._data) < 20:
            raise ValueError(f"Need at least 20 data points, got {len(self._data)}")
        return self

    def find_critical(self) -> SusceptibilityResult:
        """Compute batch-size susceptibility and find K_c.

        Returns:
            SusceptibilityResult with K_c, kappa, alpha, and raw curves.
        """
        if self._data is None:
            raise RuntimeError("No data. Call .feed() first.")
        return susceptibility_from_losses(
            self._data,
            K_min=self.K_min,
            K_max=self.K_max,
            n_points=self.n_points,
            min_batches=self.min_batches,
        )

    def find_critical_from_sweep(
        self,
        batch_sizes: Sequence[float],
        metrics: Sequence[float],
    ) -> SusceptibilityResult:
        """Find K_c from a batch-size sweep (multiple training runs).

        Args:
            batch_sizes: Batch sizes tested.
            metrics: Metric value (loss, accuracy, ...) at each batch size.

        Returns:
            SusceptibilityResult with K_c = optimal batch size.
        """
        return susceptibility_from_sweep(
            np.asarray(batch_sizes, dtype=np.float64),
            np.asarray(metrics, dtype=np.float64),
        )


def susceptibility_from_losses(
    losses: np.ndarray,
    K_min: int = 2,
    K_max: Optional[int] = None,
    n_points: int = 60,
    min_batches: int = 10,
) -> SusceptibilityResult:
    """Compute batch-size susceptibility from a 1D loss/metric sequence.

    This is the core algorithm. Given N sequential measurements:
      1. For each batch size K, partition into N//K batches
      2. Compute batch means
      3. V(K) = Var(batch means)
      4. chi(K) = |d log V / d log K|
      5. K_c = argmax(chi)

    For i.i.d. data: V(K) = sigma^2 / K, so alpha = -1.
    Deviations from alpha = -1 indicate temporal structure.

    Args:
        losses: 1D array of sequential metric values.
        K_min: Smallest batch size.
        K_max: Largest batch size (default: N // min_batches).
        n_points: Number of K values to sample.
        min_batches: Minimum batches required per K.

    Returns:
        SusceptibilityResult.
    """
    N = len(losses)

    if K_max is None:
        K_max = N // min_batches

    K_max = min(K_max, N // min_batches)
    K_min = max(K_min, 2)

    if K_max <= K_min:
        K_max = N // 2
        min_batches = 2

    # Generate K values (log-spaced, unique integers)
    K_values = np.unique(
        np.geomspace(K_min, K_max, n_points).astype(int)
    )
    K_values = K_values[(K_values >= K_min) & (K_values <= K_max)]

    # Compute V(K) at each batch size
    V = np.full(len(K_values), np.nan)
    for i, K in enumerate(K_values):
        n_batches = N // K
        if n_batches < 2:
            continue
        batches = losses[: n_batches * K].reshape(n_batches, K)
        batch_means = np.mean(batches, axis=1)
        V[i] = np.var(batch_means, ddof=1)

    # Filter valid points
    valid = np.isfinite(V) & (V > 0)
    if np.sum(valid) < 5:
        # Not enough data — return degenerate result
        return SusceptibilityResult(
            K_c=float(K_values[len(K_values) // 2]),
            kappa=1.0,
            alpha=-1.0,
            alpha_se=np.inf,
            p_iid=1.0,
            K_values=K_values,
            V_values=V,
            chi_values=np.zeros_like(V),
            diagnostics={"warning": "insufficient valid points"},
        )

    # Log-log regression: V(K) ~ K^alpha
    log_K = np.log10(K_values[valid].astype(float))
    log_V = np.log10(V[valid])
    slope, intercept, r_val, p_val, se = stats.linregress(log_K, log_V)
    alpha = slope
    alpha_se = se

    # Test H0: alpha = -1
    t_stat = (alpha - (-1.0)) / alpha_se if alpha_se > 0 else 0.0
    df = np.sum(valid) - 2
    p_iid = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(df, 1))))

    # Compute chi(K) = |d log V / d log K| via finite differences
    chi = np.full(len(K_values), np.nan)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 2:
        log_V_valid = log_V
        log_K_valid = log_K
        dlogV = np.gradient(log_V_valid, log_K_valid)
        chi[valid_idx] = np.abs(dlogV)

    # Find K_c = argmax(chi), excluding edge points
    chi_interior = chi.copy()
    chi_interior[:2] = np.nan  # skip first 2 points (edge effects)
    chi_interior[-1] = np.nan  # skip last point

    if np.any(np.isfinite(chi_interior)):
        peak_idx = np.nanargmax(chi_interior)
        K_c = float(K_values[peak_idx])

        # Compute kappa
        chi_finite = chi[np.isfinite(chi)]
        chi_max = np.nanmax(chi_interior)
        chi_median = np.median(chi_finite) if len(chi_finite) > 0 else 1.0
        kappa = float(chi_max / chi_median) if chi_median > 0 else 1.0
    else:
        K_c = float(K_values[len(K_values) // 2])
        kappa = 1.0

    # Residual analysis (deviation from power law)
    V_pred = 10 ** (intercept + alpha * np.log10(K_values.astype(float)))
    residuals = np.abs(V - V_pred) / (V_pred + 1e-30)
    residuals[~valid] = np.nan

    return SusceptibilityResult(
        K_c=K_c,
        kappa=kappa,
        alpha=alpha,
        alpha_se=alpha_se,
        p_iid=p_iid,
        K_values=K_values,
        V_values=V,
        chi_values=chi,
        diagnostics={
            "N": N,
            "n_K_tested": len(K_values),
            "n_valid": int(np.sum(valid)),
            "alpha_r2": float(r_val ** 2),
            "residual_max": float(np.nanmax(residuals)) if np.any(valid) else np.nan,
            "V_pred": V_pred,
        },
    )


def susceptibility_from_sweep(
    batch_sizes: np.ndarray,
    metrics: np.ndarray,
) -> SusceptibilityResult:
    """Find K_c from a batch-size sweep.

    Unlike susceptibility_from_losses (which re-batches a single run),
    this takes actual measurements at different batch sizes.

    Args:
        batch_sizes: Array of batch sizes used.
        metrics: Metric (loss, accuracy, ...) measured at each batch size.

    Returns:
        SusceptibilityResult.
    """
    if len(batch_sizes) != len(metrics):
        raise ValueError("batch_sizes and metrics must have same length")
    if len(batch_sizes) < 4:
        raise ValueError("Need at least 4 batch sizes for susceptibility analysis")

    # Sort by batch size
    order = np.argsort(batch_sizes)
    K = batch_sizes[order].astype(float)
    M = metrics[order].astype(float)

    # chi(K) = |dM/dK| in log space
    log_K = np.log10(K)
    chi = np.abs(np.gradient(M, log_K))

    # Find peak (skip edges)
    chi_interior = chi.copy()
    chi_interior[0] = 0
    chi_interior[-1] = 0

    peak_idx = np.argmax(chi_interior)
    K_c = float(K[peak_idx])

    chi_max = chi_interior[peak_idx]
    chi_median = np.median(chi)
    kappa = float(chi_max / chi_median) if chi_median > 0 else 1.0

    # Power-law fit on V = metric (proxy)
    valid = M > 0
    if np.sum(valid) > 3:
        slope, intercept, r_val, p_val, se = stats.linregress(
            np.log10(K[valid]), np.log10(M[valid])
        )
        alpha = slope
        alpha_se = se
        t_stat = (alpha - (-1.0)) / alpha_se if alpha_se > 0 else 0
        df = np.sum(valid) - 2
        p_iid = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(df, 1))))
    else:
        alpha, alpha_se, p_iid = np.nan, np.nan, np.nan

    return SusceptibilityResult(
        K_c=K_c,
        kappa=kappa,
        alpha=alpha,
        alpha_se=alpha_se,
        p_iid=p_iid,
        K_values=K.astype(int),
        V_values=M,
        chi_values=chi,
        diagnostics={"mode": "sweep", "n_points": len(K)},
    )
