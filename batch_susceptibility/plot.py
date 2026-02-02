# Copyright (c) 2025 ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE.txt for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Visualization for batch-size susceptibility results."""

from typing import Optional
import numpy as np

from .core import SusceptibilityResult

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    raise ImportError(
        "matplotlib is required for plotting. "
        "Install with: pip install batch-susceptibility[plot]"
    )


def plot_susceptibility(
    result: SusceptibilityResult,
    title: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> "plt.Figure":
    """Plot V(K) and chi(K) with K_c marked.

    Args:
        result: SusceptibilityResult from analysis.
        title: Optional figure title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    K = result.K_values.astype(float)
    V = result.V_values
    chi = result.chi_values

    valid = np.isfinite(V) & (V > 0)

    # Left: V(K) in log-log
    ax1.loglog(K[valid], V[valid], "o-", color="#2196F3", markersize=4, label="V(K)")

    # Power-law fit
    if "V_pred" in result.diagnostics:
        V_pred = result.diagnostics["V_pred"]
        ax1.loglog(K, V_pred, "--", color="#9E9E9E", alpha=0.7,
                   label=f"K^{result.alpha:.2f}")

    # Mark K_c
    ax1.axvline(result.K_c, color="#F44336", linestyle=":", alpha=0.8,
                label=f"K_c = {result.K_c:.0f}")

    ax1.set_xlabel("Batch size K")
    ax1.set_ylabel("V(K) = Var(batch means)")
    ax1.set_title("Variance scaling")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: chi(K)
    chi_valid = np.isfinite(chi)
    ax2.semilogx(K[chi_valid], chi[chi_valid], "s-", color="#4CAF50", markersize=4)

    ax2.axvline(result.K_c, color="#F44336", linestyle=":", alpha=0.8,
                label=f"K_c = {result.K_c:.0f}")

    # Annotate kappa
    ax2.annotate(
        f"kappa = {result.kappa:.1f}",
        xy=(result.K_c, np.nanmax(chi[chi_valid]) * 0.9),
        fontsize=10,
        color="#F44336",
        ha="center",
    )

    ax2.set_xlabel("Batch size K")
    ax2.set_ylabel("chi(K) = |d log V / d log K|")
    ax2.set_title("Susceptibility")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13)

    fig.tight_layout()
    return fig


def plot_comparison(
    results: dict,
    figsize: tuple = (14, 6),
) -> "plt.Figure":
    """Compare susceptibility across multiple experiments.

    Args:
        results: Dict of {name: SusceptibilityResult}.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, result), color in zip(results.items(), colors):
        K = result.K_values.astype(float)
        V = result.V_values
        chi = result.chi_values

        valid = np.isfinite(V) & (V > 0)
        chi_valid = np.isfinite(chi)

        ax1.loglog(K[valid], V[valid], "o-", color=color, markersize=3,
                   label=f"{name} (a={result.alpha:.2f})")

        ax2.semilogx(K[chi_valid], chi[chi_valid], "s-", color=color, markersize=3,
                     label=f"{name} (K_c={result.K_c:.0f})")

        ax2.axvline(result.K_c, color=color, linestyle=":", alpha=0.4)

    ax1.set_xlabel("Batch size K")
    ax1.set_ylabel("V(K)")
    ax1.set_title("Variance scaling")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Batch size K")
    ax2.set_ylabel("chi(K)")
    ax2.set_title("Susceptibility")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
