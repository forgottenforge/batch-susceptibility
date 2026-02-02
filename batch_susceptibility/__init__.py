# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE.txt for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Batch Susceptibility: Model-free optimal batch size finder.

Uses susceptibility analysis (chi(K) = |dV/dK|) to find the critical
batch size where training dynamics undergo a phase transition.

Based on: M.C. Wurm, "Batch-Size Susceptibility across Five Computational
Domains" (2024/2025), ForgottenForge.xyz

Example:
    >>> from batch_susceptibility import BatchSusceptibility
    >>> bs = BatchSusceptibility()
    >>> bs.feed(losses)  # per-sample or per-step losses
    >>> result = bs.find_critical()
    >>> print(f"Optimal batch size: {result.K_c}")
"""

__version__ = "1.0.0"

from .core import BatchSusceptibility, SusceptibilityResult
from .core import susceptibility_from_losses, susceptibility_from_sweep

__all__ = [
    "BatchSusceptibility",
    "SusceptibilityResult",
    "susceptibility_from_losses",
    "susceptibility_from_sweep",
]
