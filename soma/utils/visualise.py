"""
SOMA Visualise — BT/K curves, cluster visualisation.

Generates publication-ready figures for Paper 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def plot_bt_curve(
    logs: List[Dict[str, Any]],
    title: str = "SOMA — Backward Transfer over Tasks",
    save_path: Optional[str] = None,
) -> None:
    """Plot Backward Transfer across tasks.

    Args:
        logs: List of TaskLog dicts (from task_logs.jsonl).
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    tasks = [l["task_idx"] for l in logs]
    bt_values = [l["backward_transfer"] for l in logs]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(tasks, bt_values, "b-o", linewidth=2, markersize=6, label="SOMA BT")
    ax.axhline(y=-0.05, color="r", linestyle="--", linewidth=1, label="Target (BT > -0.05)")
    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Task Index", fontsize=12)
    ax.set_ylabel("Backward Transfer (BT)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_adapter_growth(
    logs: List[Dict[str, Any]],
    title: str = "SOMA — Adapter Count over Tasks",
    save_path: Optional[str] = None,
) -> None:
    """Plot adapter count K across tasks.

    Args:
        logs: List of TaskLog dicts.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    tasks = [l["task_idx"] for l in logs]
    k_values = [l["k_after"] for l in logs]
    actions = [l.get("action", "") for l in logs]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(tasks, k_values, "g-s", linewidth=2, markersize=6, label="K (adapters)")

    # Annotate actions
    for t, k, a in zip(tasks, k_values, actions):
        if "SPAWN" in a:
            ax.annotate("S", (t, k), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color="blue")
        elif "MERGE" in a:
            ax.annotate("M", (t, k), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color="red")

    ax.set_xlabel("Task Index", fontsize=12)
    ax.set_ylabel("Adapter Count (K)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_comparison(
    results: Dict[str, Dict[str, float]],
    title: str = "SOMA vs Baselines — Backward Transfer",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing methods on BT and K.

    Args:
        results: Dict mapping method name -> {'bt': float, 'k': int}.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    methods = list(results.keys())
    bt_values = [results[m]["bt"] for m in methods]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colours = ["#2196F3" if "SOMA" in m else "#9E9E9E" for m in methods]
    bars = ax.bar(methods, bt_values, color=colours, edgecolor="black", linewidth=0.5)
    ax.axhline(y=-0.05, color="r", linestyle="--", linewidth=1, label="Target")
    ax.set_ylabel("Backward Transfer (BT)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels
    for bar, val in zip(bars, bt_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def load_logs(filepath: str) -> List[Dict[str, Any]]:
    """Load task logs from a JSONL file."""
    logs = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs
