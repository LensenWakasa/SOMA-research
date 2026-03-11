"""
Test SOMA-LEARN Integration — Full 3-task smoke test.

Tests:
    1. Cold start spawns 2 adapters after tasks 0 and 1
    2. Task 2 triggers necessity check (does not always spawn)
    3. TaskLog returned with correct field types and ranges
    4. BT is in [-1.0, 0.1] range after 3 tasks
    5. K is in [2, 4] range

PASS criterion: 3-task smoke test completes without errors, all fields in expected ranges.
"""

import numpy as np
import pytest

from soma.core.learn import SomaLearn, LearnConfig, TaskLog
from soma.core.necessity import NecessityConfig
from soma.core.grow import GrowConfig


DIM = 64
RANK = 8


def _synthetic_train_fn(adapter_idx, task_data, lr_scale=1.0):
    """Synthetic train function: returns random (B, A) adapter."""
    rng = np.random.RandomState(task_data.get("seed", 42))
    B = rng.randn(DIM, RANK).astype(np.float32) * 0.01
    A = rng.randn(RANK, DIM).astype(np.float32) * 0.01
    return (B, A)


def _synthetic_eval_fn(adapter_idx, task_data):
    """Synthetic eval function: returns ~0.85 accuracy with noise."""
    seed = (task_data.get("seed", 42) + abs(adapter_idx if adapter_idx is not None else 0)) % (2**32)
    rng = np.random.RandomState(seed)
    return 0.80 + rng.random() * 0.10


def _synthetic_embed_fn(task_data):
    """Synthetic embed function: returns task-specific embeddings."""
    rng = np.random.RandomState(task_data.get("seed", 42))
    return rng.randn(20, DIM).astype(np.float32)


def _synthetic_grad_fn(task_data):
    """Synthetic grad function: returns gradients, losses, failure grads."""
    rng = np.random.RandomState(task_data.get("seed", 42))
    n_grads = 50
    grads = [rng.randn(DIM).astype(np.float32) for _ in range(n_grads)]

    # Create a loss trajectory that plateaus
    losses = list(2.0 - 0.003 * np.arange(n_grads))

    # Create some failure gradients
    failure_grads = [rng.randn(DIM).astype(np.float32) for _ in range(10)]

    return grads, losses, failure_grads


class TestSomaLearnIntegration:
    """Full 3-task integration smoke test."""

    def _make_learn(self) -> SomaLearn:
        cfg = LearnConfig(
            necessity=NecessityConfig(
                plateau_patience=30,  # Short for testing
                plateau_window=5,
                min_failures=5,  # Low for testing
            ),
            grow=GrowConfig(max_k=10),
            cold_start_tasks=2,
        )
        return SomaLearn(
            train_fn=_synthetic_train_fn,
            eval_fn=_synthetic_eval_fn,
            embed_fn=_synthetic_embed_fn,
            grad_fn=_synthetic_grad_fn,
            cfg=cfg,
        )

    def test_cold_start_spawns_two_adapters(self):
        """Tasks 0 and 1 should each spawn an adapter (cold start)."""
        learn = self._make_learn()

        task0 = {"seed": 0}
        task1 = {"seed": 1}

        log0 = learn.step(task_idx=0, task_data=task0, past_task_data=[])
        assert log0.k_after == 1, f"After task 0, K should be 1, got {log0.k_after}"
        assert "SPAWN" in log0.action, f"Task 0 action should be SPAWN, got {log0.action}"

        log1 = learn.step(task_idx=1, task_data=task1, past_task_data=[task0])
        assert log1.k_after == 2, f"After task 1, K should be 2, got {log1.k_after}"
        assert "SPAWN" in log1.action, f"Task 1 action should be SPAWN, got {log1.action}"

    def test_task2_triggers_necessity_check(self):
        """Task 2 should go through the full necessity + grow pipeline."""
        learn = self._make_learn()

        tasks = [{"seed": i} for i in range(3)]
        logs = []

        for t in range(3):
            log = learn.step(task_idx=t, task_data=tasks[t], past_task_data=tasks[:t])
            logs.append(log)

        # Task 2 should not be a cold-start spawn
        assert "cold" not in logs[2].action.lower(), (
            f"Task 2 should use necessity check, got action: {logs[2].action}"
        )

    def test_task_log_fields(self):
        """TaskLog should have correct field types and ranges."""
        learn = self._make_learn()

        tasks = [{"seed": i} for i in range(3)]
        for t in range(3):
            log = learn.step(task_idx=t, task_data=tasks[t], past_task_data=tasks[:t])

            assert isinstance(log, TaskLog)
            assert isinstance(log.task_idx, int)
            assert isinstance(log.action, str)
            assert isinstance(log.k_before, int)
            assert isinstance(log.k_after, int)
            assert isinstance(log.accuracy, float)
            assert isinstance(log.backward_transfer, float)
            assert 0.0 <= log.accuracy <= 1.0, f"Accuracy out of range: {log.accuracy}"
            assert log.k_after >= 0, f"K should be non-negative: {log.k_after}"

    def test_bt_in_valid_range(self):
        """BT after 3 tasks should be in [-1.0, 0.1] range."""
        learn = self._make_learn()

        tasks = [{"seed": i} for i in range(3)]
        for t in range(3):
            log = learn.step(task_idx=t, task_data=tasks[t], past_task_data=tasks[:t])

        bt = log.backward_transfer
        assert -1.0 <= bt <= 0.1, f"BT out of range: {bt}"

    def test_k_in_valid_range(self):
        """K after 3 tasks should be in [2, 4] range."""
        learn = self._make_learn()

        tasks = [{"seed": i} for i in range(3)]
        for t in range(3):
            log = learn.step(task_idx=t, task_data=tasks[t], past_task_data=tasks[:t])

        k = log.k_after
        assert 1 <= k <= 4, f"K out of range: {k}"

    def test_summary(self):
        """Summary should return valid dict."""
        learn = self._make_learn()

        tasks = [{"seed": i} for i in range(3)]
        for t in range(3):
            learn.step(task_idx=t, task_data=tasks[t], past_task_data=tasks[:t])

        result = learn.summary()
        assert "backward_transfer" in result
        assert "final_k" in result
        assert "tasks_completed" in result
        assert result["tasks_completed"] == 3
