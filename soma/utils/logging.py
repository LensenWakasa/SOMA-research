"""
SOMA Logging — wandb integration + JSON logging.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class SomaLogger:
    """Unified logger supporting JSON files and optional wandb integration.

    Usage::

        logger = SomaLogger(log_dir="outputs/exp1", use_wandb=False)
        logger.log({"task": 1, "bt": -0.03, "k": 3})
        logger.finish()
    """

    def __init__(
        self,
        log_dir: str = "outputs",
        use_wandb: bool = False,
        wandb_project: str = "soma",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self._wandb_run = None
        self._log_entries: list = []

        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=wandb_config or {},
                )
            except ImportError:
                print("WARNING: wandb not installed. Falling back to JSON only.")
                self.use_wandb = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics.

        Args:
            data: Key-value pairs to log.
            step: Optional step number for wandb.
        """
        # Sanitise numpy types for JSON
        sanitised = {}
        for k, v in data.items():
            if isinstance(v, (np.integer,)):
                sanitised[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sanitised[k] = float(v)
            elif isinstance(v, np.ndarray):
                sanitised[k] = v.tolist()
            else:
                sanitised[k] = v

        self._log_entries.append(sanitised)

        # Write to JSONL file
        filepath = self.log_dir / "metrics.jsonl"
        with open(filepath, "a") as f:
            f.write(json.dumps(sanitised) + "\n")

        # wandb
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.log(sanitised, step=step)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """Save a final summary JSON file."""
        filepath = self.log_dir / "summary.json"
        # Sanitise
        sanitised = {}
        for k, v in summary.items():
            if isinstance(v, (np.integer,)):
                sanitised[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sanitised[k] = float(v)
            elif isinstance(v, np.ndarray):
                sanitised[k] = v.tolist()
            else:
                sanitised[k] = v

        with open(filepath, "w") as f:
            json.dump(sanitised, f, indent=2)

    def finish(self) -> None:
        """Finish logging session."""
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.finish()
