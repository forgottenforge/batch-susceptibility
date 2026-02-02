# Copyright (c) ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE.txt for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Command-line interface for batch-size susceptibility.

Usage:
    # From a CSV of per-step losses:
    batch-susceptibility losses.csv --column loss

    # From a batch-size sweep CSV:
    batch-susceptibility sweep.csv --batch-col batch_size --metric-col loss --mode sweep

    # Pipe from stdin:
    cat losses.txt | batch-susceptibility -
"""

import argparse
import sys
import csv
from typing import List, Optional
import numpy as np

from .core import BatchSusceptibility, SusceptibilityResult


def read_values_from_file(path: str, column: Optional[str] = None) -> np.ndarray:
    """Read numeric values from a file (CSV, TSV, or plain text)."""
    if path == "-":
        lines = sys.stdin.readlines()
    else:
        with open(path, "r") as f:
            lines = f.readlines()

    if not lines:
        raise ValueError("Empty input")

    # Detect CSV
    first_line = lines[0].strip()
    if "," in first_line or "\t" in first_line:
        delimiter = "," if "," in first_line else "\t"
        reader = csv.DictReader(lines, delimiter=delimiter)
        rows = list(reader)

        if column:
            if column not in rows[0]:
                available = ", ".join(rows[0].keys())
                raise ValueError(
                    f"Column '{column}' not found. Available: {available}"
                )
            values = [float(r[column]) for r in rows if r[column].strip()]
        else:
            # Use last numeric column
            for key in reversed(list(rows[0].keys())):
                try:
                    float(rows[0][key])
                    column = key
                    break
                except (ValueError, TypeError):
                    continue
            if column is None:
                raise ValueError("No numeric column found")
            values = [float(r[column]) for r in rows if r[column].strip()]
    else:
        # Plain text: one number per line
        values = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    values.append(float(line))
                except ValueError:
                    continue

    return np.array(values, dtype=np.float64)


def read_sweep_from_file(
    path: str,
    batch_col: str,
    metric_col: str,
) -> tuple:
    """Read batch-size sweep data from CSV."""
    with open(path, "r") as f:
        lines = f.readlines()

    delimiter = "," if "," in lines[0] else "\t"
    reader = csv.DictReader(lines, delimiter=delimiter)
    rows = list(reader)

    batch_sizes = np.array([float(r[batch_col]) for r in rows])
    metrics = np.array([float(r[metric_col]) for r in rows])
    return batch_sizes, metrics


def format_result(result: SusceptibilityResult, verbose: bool = False) -> str:
    """Format result for terminal output."""
    lines = [
        "=" * 60,
        "  BATCH-SIZE SUSCEPTIBILITY ANALYSIS",
        "=" * 60,
        "",
        result.summary(),
        "",
    ]

    if result.is_significant:
        lines.append(f"  >> RECOMMENDATION: Use batch size {result.K_c:.0f}")
    else:
        lines.append(f"  >> No significant critical scale found (kappa={result.kappa:.2f})")
        lines.append(f"     Your data appears {result.regime}.")

    if verbose:
        lines.extend([
            "",
            "  K            V(K)           chi(K)",
            "  " + "-" * 45,
        ])
        for i in range(len(result.K_values)):
            v_str = f"{result.V_values[i]:.6e}" if np.isfinite(result.V_values[i]) else "NaN"
            c_str = f"{result.chi_values[i]:.4f}" if np.isfinite(result.chi_values[i]) else "NaN"
            marker = " <<" if result.K_values[i] == result.K_c else ""
            lines.append(f"  {result.K_values[i]:<10d} {v_str:>14s} {c_str:>12s}{marker}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        prog="batch-susceptibility",
        description="Find optimal batch size using susceptibility analysis.",
    )
    parser.add_argument(
        "input",
        help="Input file (CSV, TSV, or plain text). Use '-' for stdin.",
    )
    parser.add_argument(
        "--mode",
        choices=["online", "sweep"],
        default="online",
        help="Analysis mode: 'online' for per-step losses, 'sweep' for batch-size sweep.",
    )
    parser.add_argument(
        "--column", "-c",
        help="Column name for metric values (CSV mode).",
    )
    parser.add_argument(
        "--batch-col",
        help="Column name for batch sizes (sweep mode).",
    )
    parser.add_argument(
        "--metric-col",
        help="Column name for metric values (sweep mode).",
    )
    parser.add_argument(
        "--K-min",
        type=int,
        default=2,
        help="Minimum batch size to test (online mode).",
    )
    parser.add_argument(
        "--K-max",
        type=int,
        default=None,
        help="Maximum batch size to test (online mode).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including V(K) table.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="susceptibility.png",
        help="Save plot to file (requires matplotlib).",
    )

    args = parser.parse_args()

    if args.mode == "sweep":
        if not args.batch_col or not args.metric_col:
            parser.error("Sweep mode requires --batch-col and --metric-col")
        batch_sizes, metrics = read_sweep_from_file(
            args.input, args.batch_col, args.metric_col
        )
        bs = BatchSusceptibility()
        result = bs.find_critical_from_sweep(batch_sizes, metrics)
    else:
        values = read_values_from_file(args.input, args.column)
        bs = BatchSusceptibility(K_min=args.K_min, K_max=args.K_max)
        bs.feed(values)
        result = bs.find_critical()

    if args.json:
        import json
        out = {
            "K_c": result.K_c,
            "kappa": result.kappa,
            "alpha": result.alpha,
            "alpha_se": result.alpha_se,
            "p_iid": result.p_iid,
            "regime": result.regime,
            "significant": result.is_significant,
        }
        print(json.dumps(out, indent=2))
    else:
        print(format_result(result, verbose=args.verbose))

    if args.plot:
        try:
            from .plot import plot_susceptibility
            fig = plot_susceptibility(result)
            fig.savefig(args.plot, dpi=150, bbox_inches="tight")
            print(f"\n  Plot saved to: {args.plot}")
        except ImportError:
            print("\n  Warning: matplotlib required for plotting. "
                  "pip install batch-susceptibility[plot]")


if __name__ == "__main__":
    main()
