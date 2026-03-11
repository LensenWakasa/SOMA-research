"""
SOMA — Compare Results across methods.

Usage:
    python -m soma.utils.compare_results --runs soma ewc replay sequential
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def load_result(method: str) -> Dict:
    """Load results JSON for a method."""
    paths = [
        Path("outputs") / "permuted_mnist" / f"results.json",
        Path("outputs") / "baselines" / f"{method}_results.json",
        Path("outputs") / f"{method}_results.json",
    ]

    if method == "soma":
        paths = [Path("outputs") / "permuted_mnist" / "results.json"] + paths

    for p in paths:
        if p.exists():
            with open(p) as f:
                return json.load(f)

    print(f"WARNING: No results found for '{method}'")
    return {}


def compare(methods: list) -> None:
    """Compare BT and K across methods."""
    print("\n" + "=" * 60)
    print("SOMA vs Baselines — Results Comparison")
    print("=" * 60)
    print(f"{'Method':<20} {'BT':>10} {'K':>6} {'Acc':>8}")
    print("-" * 60)

    for method in methods:
        result = load_result(method)
        bt = result.get("backward_transfer", "N/A")
        k = result.get("final_k", "N/A")
        acc = result.get("final_accuracy", "N/A")

        if isinstance(bt, float):
            bt_str = f"{bt:.4f}"
        else:
            bt_str = str(bt)

        if isinstance(acc, float):
            acc_str = f"{acc:.3f}"
        else:
            acc_str = str(acc)

        print(f"{method:<20} {bt_str:>10} {str(k):>6} {acc_str:>8}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare SOMA vs baselines")
    parser.add_argument("--runs", nargs="+", default=["soma", "ewc", "replay", "sequential"])
    args = parser.parse_args()
    compare(args.runs)


if __name__ == "__main__":
    main()
