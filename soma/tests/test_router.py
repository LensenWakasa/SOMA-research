"""
Test Router — Prototype-based adapter routing.

Tests:
    1. Route task 0 examples to adapter 0 (>80% accuracy)
    2. Route task 1 examples to adapter 1 (>80% accuracy)
    3. Add adapter 2; re-route task 0 → still routes to adapter 0 (no forgetting)
    4. Remove adapter; verify re-indexing works
    5. Empty router raises error

PASS criterion: Routing accuracy >80% for all adapters, including after new adapters added.
"""

import numpy as np
import pytest

from soma.core.router import SomaRouter


DIM = 64


def _make_task_embeddings(direction: np.ndarray, n: int = 50, noise: float = 0.1) -> np.ndarray:
    """Generate embeddings clustered around a direction."""
    rng = np.random.RandomState(42)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    embs = direction[np.newaxis, :] + rng.randn(n, DIM) * noise
    return embs.astype(np.float32)


class TestSomaRouter:
    """Router: prototype-based routing, no forgetting."""

    def test_routes_task0_correctly(self):
        """Task 0 examples should route to adapter 0."""
        router = SomaRouter(n_prototypes=10)

        dir0 = np.zeros(DIM); dir0[0] = 1.0
        dir1 = np.zeros(DIM); dir1[1] = 1.0

        router.register(0, _make_task_embeddings(dir0))
        router.register(1, _make_task_embeddings(dir1))

        # Route task 0 examples
        test_embs = _make_task_embeddings(dir0, n=20, noise=0.1)
        correct = sum(1 for e in test_embs if router.route(e)[0] == 0)
        accuracy = correct / len(test_embs)
        assert accuracy > 0.80, f"Routing accuracy for task 0: {accuracy:.2f} < 0.80"

    def test_routes_task1_correctly(self):
        """Task 1 examples should route to adapter 1."""
        router = SomaRouter(n_prototypes=10)

        dir0 = np.zeros(DIM); dir0[0] = 1.0
        dir1 = np.zeros(DIM); dir1[1] = 1.0

        router.register(0, _make_task_embeddings(dir0))
        router.register(1, _make_task_embeddings(dir1))

        test_embs = _make_task_embeddings(dir1, n=20, noise=0.1)
        correct = sum(1 for e in test_embs if router.route(e)[0] == 1)
        accuracy = correct / len(test_embs)
        assert accuracy > 0.80, f"Routing accuracy for task 1: {accuracy:.2f} < 0.80"

    def test_no_forgetting_after_adding_adapter(self):
        """Adding adapter 2 should NOT change routing for task 0."""
        router = SomaRouter(n_prototypes=10)

        dir0 = np.zeros(DIM); dir0[0] = 1.0
        dir1 = np.zeros(DIM); dir1[1] = 1.0
        dir2 = np.zeros(DIM); dir2[2] = 1.0

        router.register(0, _make_task_embeddings(dir0))
        router.register(1, _make_task_embeddings(dir1))

        # Check task 0 routing before adding adapter 2
        test_embs = _make_task_embeddings(dir0, n=20, noise=0.1)
        before_correct = sum(1 for e in test_embs if router.route(e)[0] == 0)

        # Add adapter 2
        router.register(2, _make_task_embeddings(dir2))

        # Check task 0 routing after adding adapter 2
        after_correct = sum(1 for e in test_embs if router.route(e)[0] == 0)
        accuracy = after_correct / len(test_embs)

        assert accuracy > 0.80, f"Routing accuracy after adding adapter: {accuracy:.2f} < 0.80"
        assert after_correct >= before_correct - 1, (
            "Routing should not degrade after adding a new adapter"
        )

    def test_remove_and_reindex(self):
        """Removing an adapter should re-index remaining adapters."""
        router = SomaRouter(n_prototypes=10)

        dir0 = np.zeros(DIM); dir0[0] = 1.0
        dir1 = np.zeros(DIM); dir1[1] = 1.0
        dir2 = np.zeros(DIM); dir2[2] = 1.0

        router.register(0, _make_task_embeddings(dir0))
        router.register(1, _make_task_embeddings(dir1))
        router.register(2, _make_task_embeddings(dir2))

        assert router.n_adapters == 3

        router.remove(1)  # Remove middle adapter
        assert router.n_adapters == 2
        # Remaining adapters should be re-indexed to 0 and 1
        assert 0 in router._prototypes
        assert 1 in router._prototypes

    def test_empty_router_raises(self):
        """Routing with no adapters should raise RuntimeError."""
        router = SomaRouter(n_prototypes=10)
        with pytest.raises(RuntimeError):
            router.route(np.random.randn(DIM))

    def test_confidence_score(self):
        """Route confidence should be reasonable (0-1 range)."""
        router = SomaRouter(n_prototypes=10)
        dir0 = np.zeros(DIM); dir0[0] = 1.0
        router.register(0, _make_task_embeddings(dir0))

        query = np.zeros(DIM); query[0] = 1.0
        idx, conf = router.route(query)
        assert idx == 0
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"
        assert conf > 0.5, f"Confidence for matching query should be high: {conf}"
