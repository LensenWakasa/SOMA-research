"""
SOMA-LEARN — Outer training loop.

Orchestrates all SOMA components across the task stream:
    1. SomaNecessity — detects when new capacity is genuinely needed
    2. SomaGrow — RL controller selects growth action
    3. SomaRouter — prototype-based adapter routing

Handles cold start (first 2 tasks always spawn), enforces growth ceiling,
computes BT and FT metrics at each eval interval, and writes JSON logs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from soma.core.necessity import SomaNecessity, NecessityConfig, NecessityResult
from soma.core.grow import SomaGrow, GrowConfig, GrowAction, GrowResult
from soma.core.router import SomaRouter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LearnConfig:
    """Configuration for the SOMA-LEARN outer loop."""

    # Sub-component configs
    necessity: NecessityConfig = field(default_factory=NecessityConfig)
    grow: GrowConfig = field(default_factory=GrowConfig)

    # Router
    n_prototypes: int = 10
    """Prototypes per adapter for routing."""

    # Cold start
    cold_start_tasks: int = 2
    """Number of initial tasks that always spawn (no necessity check)."""

    # Logging
    log_dir: Optional[str] = None
    """Directory for JSON logs. None = no logging."""

    # Device (informational — actual device management in train/eval fns)
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Task log
# ---------------------------------------------------------------------------


@dataclass
class TaskLog:
    """Log entry for one task in the SOMA-LEARN loop."""

    task_idx: int
    action: str
    k_before: int
    k_after: int
    accuracy: float
    backward_transfer: float
    forward_transfer: float
    reward: float
    necessity: bool = False
    n1: bool = False
    n2: bool = False
    n3: bool = False
    plateau_score: float = 0.0
    residual_fraction: float = 0.0
    silhouette: float = 0.0
    entropy: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# SomaLearn — Outer Loop
# ---------------------------------------------------------------------------


class SomaLearn:
    """Complete SOMA outer training loop.

    Orchestrates necessity detection, growth decisions, adapter management,
    routing, and metric computation across a sequential task stream.

    Usage::

        learn = SomaLearn(train_fn, eval_fn, embed_fn)
        past_tasks = []
        for t, task in enumerate(task_stream):
            log = learn.step(task_idx=t, task_data=task, past_task_data=past_tasks)
            past_tasks.append(task)
            print(f'BT: {log.backward_transfer:.4f}  K: {log.k_after}')
        learn.summary()

    Args:
        train_fn: ``train_fn(adapter_idx_or_None, task_data, lr_scale) -> (B, A)``
            * adapter_idx=None -> spawn new adapter
            * adapter_idx=int -> fine-tune existing adapter
        eval_fn: ``eval_fn(adapter_idx, task_data) -> accuracy``
            * adapter_idx=-1 -> use best available adapter
        embed_fn: ``embed_fn(task_data) -> np.ndarray [n, embed_dim]``
        grad_fn: ``grad_fn(task_data) -> list[np.ndarray]``
            Returns list of per-sample gradient vectors for necessity signals.
        cfg: :class:`LearnConfig`.
    """

    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        embed_fn: Callable,
        grad_fn: Optional[Callable] = None,
        cfg: Optional[LearnConfig] = None,
    ) -> None:
        self.cfg = cfg or LearnConfig()
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.embed_fn = embed_fn
        self.grad_fn = grad_fn

        # Components
        self.necessity = SomaNecessity(self.cfg.necessity)
        self.grow = SomaGrow(self.cfg.grow)
        self.router = SomaRouter(n_prototypes=self.cfg.n_prototypes)

        # Adapter pool: list of (B, A) tuples
        self.pool: List[tuple] = []

        # Tracking
        self.history: List[TaskLog] = []
        self._peak_accuracies: Dict[int, float] = {}  # task_idx -> peak acc (diagonal)
        self._acc_matrix: List[List[float]] = []      # [task_t][task_i] = acc on i after t
        self._steps_since_spawn: int = 0
        self._spawn_count: int = 0
        self._merge_count: int = 0

    @property
    def k(self) -> int:
        """Current adapter count."""
        return len(self.pool)

    def step(
        self,
        task_idx: int,
        task_data: Any,
        past_task_data: List[Any],
    ) -> TaskLog:
        """Process one task in the SOMA-LEARN loop.

        Args:
            task_idx: Sequential task index (0-based).
            task_data: Data for the current task.
            past_task_data: List of data for all previously completed tasks.

        Returns:
            :class:`TaskLog` with metrics and action details.
        """
        t0 = time.time()

        # ------------------------------------------------------------------
        # COLD START: first N tasks always spawn unconditionally
        # ------------------------------------------------------------------
        if task_idx < self.cfg.cold_start_tasks:
            return self._cold_start_step(task_idx, task_data, past_task_data, t0)

        # ------------------------------------------------------------------
        # MAIN LOOP: necessity check + RL-guided growth
        # ------------------------------------------------------------------
        self.necessity.reset_for_task()

        # Feed signals to necessity detector
        grads_this_task = []
        if self.grad_fn is not None:
            grads, loss_values, failure_grads = self.grad_fn(task_data)
            for loss_val in loss_values:
                self.necessity.update_loss(loss_val)
            for g in grads:
                self.necessity.add_gradient(g)
                grads_this_task.append(g)
            for fg in failure_grads:
                self.necessity.add_failure_gradient(fg)
        else:
            # If no grad_fn, simulate with training signals
            # The experiment scripts should provide a proper grad_fn
            pass

        # Check necessity
        nec_result = self.necessity.check()

        # Build RL state vector
        nec_features = self.necessity.rl_state_features()
        router_conf = 0.0
        if self.router.n_adapters > 0 and self.embed_fn is not None:
            embs = self.embed_fn(task_data)
            if len(embs) > 0:
                router_conf = self.router.max_confidence(embs[0])

        state = self.grow.build_state_vector(
            nec_features=nec_features,
            k=self.k,
            max_k=self.cfg.grow.max_k,
            router_confidence=router_conf,
            steps_since_spawn=self._steps_since_spawn,
        )

        # Determine forced action: ceiling -> MERGE; warmup -> unconditional SPAWN
        force_action = None
        if self.k >= self.cfg.grow.max_k:
            force_action = GrowAction.MERGE
        elif task_idx < self.cfg.cold_start_tasks + 2:
            # Paper 1 warmup: force SPAWN for the first 2 post-cold-start tasks
            # so each early task receives its own clean adapter.
            force_action = GrowAction.SPAWN_NEW

        # SOMA-GROW step
        grow_result = self.grow.step(
            state=state,
            nec_result=nec_result,
            pool=self.pool,
            task_data=task_data,
            past_task_data=past_task_data,
            train_fn=self.train_fn,
            eval_fn=self.eval_fn,
            force_action=force_action,
        )

        # Update router if spawn occurred
        if grow_result.action == GrowAction.SPAWN_NEW:
            embs = self.embed_fn(task_data)
            self.router.register(self.k - 1, embs)
            self._steps_since_spawn = 0
            self._spawn_count += 1
        elif grow_result.action == GrowAction.MERGE:
            self._merge_count += 1
        else:
            self._steps_since_spawn += 1

        # Update necessity subspace basis
        self.necessity.task_completed(grads_this_task)

        # Policy update
        if self.grow.should_update_policy():
            self.grow.update_policy()

        # Compute metrics
        acc = self.eval_fn(-1, task_data) if self.k > 0 else 0.0
        self._peak_accuracies[task_idx] = acc
        self._record_acc_row(task_idx, past_task_data, task_data)
        bt = self._compute_bt(task_idx)
        ft = self._compute_ft(task_idx, task_data)

        elapsed = time.time() - t0

        log = TaskLog(
            task_idx=task_idx,
            action=grow_result.action_name or grow_result.action.name,
            k_before=grow_result.k_before,
            k_after=grow_result.k_after,
            accuracy=acc,
            backward_transfer=bt,
            forward_transfer=ft,
            reward=grow_result.reward,
            necessity=nec_result.necessity,
            n1=nec_result.n1,
            n2=nec_result.n2,
            n3=nec_result.n3,
            plateau_score=nec_result.plateau_score,
            residual_fraction=nec_result.residual_fraction,
            silhouette=nec_result.silhouette,
            entropy=nec_result.entropy,
            elapsed_seconds=elapsed,
        )

        self.history.append(log)
        self._write_log(log)
        return log

    def _cold_start_step(
        self, task_idx: int, task_data: Any, past_task_data: List[Any], t0: float
    ) -> TaskLog:
        """Cold-start: always spawn new adapter for task."""
        k_before = self.k

        # Spawn
        B, A = self.train_fn(None, task_data, 1.0)
        self.pool.append((B, A))
        self._spawn_count += 1

        # Register with router
        embs = self.embed_fn(task_data)
        self.router.register(self.k - 1, embs)

        # Feed necessity calibration
        self.necessity.reset_for_task()
        grads_this_task = []
        if self.grad_fn is not None:
            grads, loss_values, failure_grads = self.grad_fn(task_data)
            for loss_val in loss_values:
                self.necessity.update_loss(loss_val)
            for g in grads:
                self.necessity.add_gradient(g)
                grads_this_task.append(g)
            for fg in failure_grads:
                self.necessity.add_failure_gradient(fg)
        self.necessity.task_completed(grads_this_task)

        # Metrics
        acc = self.eval_fn(-1, task_data) if self.k > 0 else 0.0
        self._peak_accuracies[task_idx] = acc
        self._record_acc_row(task_idx, past_task_data, task_data)
        bt = self._compute_bt(task_idx)
        ft = 0.0  # No forward transfer for cold start

        elapsed = time.time() - t0

        log = TaskLog(
            task_idx=task_idx,
            action="SPAWN(cold)",
            k_before=k_before,
            k_after=self.k,
            accuracy=acc,
            backward_transfer=bt,
            forward_transfer=ft,
            reward=0.0,
            elapsed_seconds=elapsed,
        )

        self.history.append(log)
        self._write_log(log)
        return log

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _record_acc_row(self, task_idx: int, past_task_data: List[Any], task_data: Any) -> None:
        """Record a row in the accuracy matrix: acc on every seen task after task_idx.

        Row T stores [acc(T,0), acc(T,1), ..., acc(T,T)] where acc(T,i) is the
        model's accuracy on task i immediately after finishing task T.
        """
        if self.k == 0:
            self._acc_matrix.append([])
            return
        row = [self.eval_fn(-1, d) for d in past_task_data]
        row.append(self.eval_fn(-1, task_data))  # current task
        self._acc_matrix.append(row)

    def _compute_bt(self, task_idx: int) -> float:
        """Backward Transfer using accumulated accuracy matrix.

        BT = (1 / T) * sum_{i=0}^{T-1} [acc_matrix[T][i] - acc_matrix[i][i]]

        Where acc_matrix[t][i] = accuracy on task i right after training on task t.
        The diagonal acc_matrix[i][i] is the peak accuracy on task i.
        """
        T = task_idx  # number of past tasks
        if T == 0 or len(self._acc_matrix) < T + 1:
            return 0.0

        current_row = self._acc_matrix[task_idx]  # acc on all tasks after training on T
        diffs = []
        for i in range(T):
            if i >= len(current_row):
                break
            acc_now = current_row[i]
            # Diagonal: acc on task i right after training on i
            if i < len(self._acc_matrix) and i < len(self._acc_matrix[i]):
                acc_peak = self._acc_matrix[i][i]
            else:
                acc_peak = self._peak_accuracies.get(i, acc_now)
            diffs.append(acc_now - acc_peak)

        return float(np.mean(diffs)) if diffs else 0.0

    def _compute_ft(self, task_idx: int, task_data: Any) -> float:
        """Forward Transfer: how much past learning helps on new task.

        FT = A(M_{i-1}, T_i) - A(random, T_i)
        For simplicity, A(random) = 0.10 for 10-class problems.
        """
        if task_idx == 0 or self.k == 0:
            return 0.0
        # We measure accuracy before training on the task
        # In practice, the eval_fn should give pre-training accuracy
        # For now, return 0.0 (experiments will compute FT properly)
        return 0.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Print and return final metrics summary."""
        if len(self.history) == 0:
            print("No tasks completed.")
            return {}

        last = self.history[-1]
        result = {
            "backward_transfer": last.backward_transfer,
            "forward_transfer": last.forward_transfer,
            "final_k": last.k_after,
            "spawn_count": self._spawn_count,
            "merge_count": self._merge_count,
            "tasks_completed": len(self.history),
            "target_bt_met": last.backward_transfer > -0.05,
        }

        print("\n=== SOMA Summary ===")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v}")

        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_log(self, log: TaskLog) -> None:
        """Append task log to JSON file."""
        if self.cfg.log_dir is None:
            return

        log_path = Path(self.cfg.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        filepath = log_path / "task_logs.jsonl"

        with open(filepath, "a") as f:
            f.write(json.dumps(log.to_dict()) + "\n")
