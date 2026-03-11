"""
SOMA Checkpoint — Save/load adapter pool + policy weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def save_checkpoint(
    filepath: str,
    pool: List[Tuple[np.ndarray, np.ndarray]],
    policy_weights: Dict[str, np.ndarray],
    router_prototypes: Dict[int, np.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save SOMA system state to a .npz checkpoint.

    Args:
        filepath: Path for the checkpoint file (e.g., 'checkpoints/soma_task5.npz').
        pool: List of (B, A) adapter tuples.
        policy_weights: Dict with 'W' and 'b' arrays from GrowthPolicy.
        router_prototypes: Dict mapping adapter_idx -> prototypes array.
        metadata: Optional dict with additional info (saved as JSON string).
    """
    save_dict: Dict[str, Any] = {}

    # Save adapter pool
    save_dict["n_adapters"] = np.array([len(pool)])
    for i, (B, A) in enumerate(pool):
        save_dict[f"pool_B_{i}"] = B
        save_dict[f"pool_A_{i}"] = A

    # Save policy weights
    save_dict["policy_W"] = policy_weights["W"]
    save_dict["policy_b"] = policy_weights["b"]

    # Save router prototypes
    save_dict["n_router_entries"] = np.array([len(router_prototypes)])
    for idx, protos in router_prototypes.items():
        save_dict[f"router_{idx}"] = protos

    # Save metadata as JSON string in a special array
    if metadata is not None:
        json_str = json.dumps(metadata)
        save_dict["metadata"] = np.array([json_str])

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **save_dict)


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """Load SOMA system state from a .npz checkpoint.

    Returns:
        Dict with keys:
            - 'pool': List of (B, A) tuples
            - 'policy_weights': Dict with 'W' and 'b'
            - 'router_prototypes': Dict mapping int -> np.ndarray
            - 'metadata': Optional dict
    """
    data = np.load(filepath, allow_pickle=True)

    # Load adapter pool
    n_adapters = int(data["n_adapters"][0])
    pool = []
    for i in range(n_adapters):
        B = data[f"pool_B_{i}"]
        A = data[f"pool_A_{i}"]
        pool.append((B, A))

    # Load policy weights
    policy_weights = {
        "W": data["policy_W"],
        "b": data["policy_b"],
    }

    # Load router prototypes
    n_router = int(data["n_router_entries"][0])
    router_prototypes = {}
    for idx in range(n_router):
        key = f"router_{idx}"
        if key in data:
            router_prototypes[idx] = data[key]

    # Load metadata
    metadata = None
    if "metadata" in data:
        json_str = str(data["metadata"][0])
        metadata = json.loads(json_str)

    return {
        "pool": pool,
        "policy_weights": policy_weights,
        "router_prototypes": router_prototypes,
        "metadata": metadata,
    }
