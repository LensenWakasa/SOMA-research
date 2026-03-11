"""
Test N3 — Systematic Failure Detection.

Tests:
    1. Systematic set A: 50 near-parallel gradients -> N3=True
    2. Random set B: 50 random-direction gradients -> N3=False
    3. Small set C: only 10 failures -> N3=False (insufficient data)
    4. Calibration: feed random set as cold-start, verify threshold = 0.70 * baseline entropy

PASS criterion: N3 correctly classifies all 4 synthetic test cases.
"""

import numpy as np
import pytest

from soma.core.necessity import SomaNecessity, NecessityConfig


DIM = 128  # Larger dim for better clustering


class TestN3SystematicFailure:
    """N3: verify fires on systematic failures, not random noise."""

    def test_systematic_parallel_gradients(self):
        """Near-parallel failure gradients should trigger N3=True.

        All failures need ~same fix -> gradients point in ~same direction.
        When split into 2+ clusters (via small perturbation), they form
        tight clusters with high silhouette and low entropy.
        """
        cfg = NecessityConfig(min_failures=25, proj_dim=32, silhouette_min=0.30)
        nec = SomaNecessity(cfg)
        nec.reset_for_task()

        rng = np.random.RandomState(42)

        # Create 2 tight clusters of failure gradients
        # Cluster 1: around e_0
        direction_1 = np.zeros(DIM)
        direction_1[0] = 1.0
        # Cluster 2: around e_1
        direction_2 = np.zeros(DIM)
        direction_2[1] = 1.0

        for i in range(30):
            g = direction_1 + rng.randn(DIM) * 0.05  # tight cluster
            nec.add_failure_gradient(g)

        for i in range(30):
            g = direction_2 + rng.randn(DIM) * 0.05  # tight cluster
            nec.add_failure_gradient(g)

        n3, sil, entropy, n_fail = nec._check_n3()
        assert n_fail == 60
        assert sil > 0.30, f"Silhouette should be >0.30 for tight clusters, got {sil:.4f}"
        # n3 depends on entropy threshold — with default 0.70 it should fire
        # The two clusters have low entropy
        assert n3, f"N3 should be True for systematic parallel failures (sil={sil:.3f}, ent={entropy:.3f})"

    def test_random_gradients_no_clustering(self):
        """Random failure gradients should NOT trigger N3.

        Failures need different fixes -> no cluster structure.
        """
        cfg = NecessityConfig(min_failures=25, proj_dim=32, silhouette_min=0.30)
        nec = SomaNecessity(cfg)
        nec.reset_for_task()

        rng = np.random.RandomState(123)
        for _ in range(50):
            g = rng.randn(DIM)  # random directions
            nec.add_failure_gradient(g)

        n3, sil, entropy, n_fail = nec._check_n3()
        assert n_fail == 50
        # Random gradients should not form meaningful clusters
        # Either silhouette < 0.30 or DBSCAN finds no clusters
        assert not n3, f"N3 should be False for random failures (sil={sil:.3f}, ent={entropy:.3f})"

    def test_insufficient_failures(self):
        """Fewer than min_failures should return N3=False."""
        cfg = NecessityConfig(min_failures=25)
        nec = SomaNecessity(cfg)
        nec.reset_for_task()

        rng = np.random.RandomState(42)
        for _ in range(10):  # only 10 < 25
            g = rng.randn(DIM)
            nec.add_failure_gradient(g)

        n3, sil, entropy, n_fail = nec._check_n3()
        assert n_fail == 10
        assert not n3, "N3 should be False with insufficient failures"
        assert sil == 0.0, "Silhouette should be 0.0 with insufficient data"

    def test_entropy_calibration(self):
        """Calibration from cold-start tasks should set threshold = 0.70 * mean entropy."""
        cfg = NecessityConfig(
            min_failures=25,
            proj_dim=32,
            calibration_tasks=2,
            entropy_default=0.70,
        )
        nec = SomaNecessity(cfg)

        rng = np.random.RandomState(42)

        # Simulate 2 cold-start tasks with random gradients
        for task in range(2):
            nec.reset_for_task()
            grads = []
            # Create two-cluster structure so entropy is computable
            direction_a = np.zeros(DIM)
            direction_a[task * 2] = 1.0
            direction_b = np.zeros(DIM)
            direction_b[task * 2 + 1] = 1.0
            for _ in range(20):
                g = direction_a + rng.randn(DIM) * 0.05
                nec.add_failure_gradient(g)
                grads.append(g)
            for _ in range(20):
                g = direction_b + rng.randn(DIM) * 0.05
                nec.add_failure_gradient(g)
                grads.append(g)

            nec.task_completed(grads)

        # After 2 calibration tasks, threshold should be set
        assert nec._entropy_threshold is not None, "Entropy threshold should be calibrated"
        assert nec._entropy_threshold > 0, f"Threshold should be positive: {nec._entropy_threshold}"
        # Threshold = 0.70 * mean of the two calibration entropies
        expected = 0.70 * np.mean(nec._calibration_entropies)
        assert abs(nec._entropy_threshold - expected) < 1e-6, (
            f"Threshold {nec._entropy_threshold:.4f} != 0.70 * {np.mean(nec._calibration_entropies):.4f}"
        )
