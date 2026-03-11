"""
Test N2 — Subspace Saturation Detector.

Tests:
    1. Gradient in subspace (e_1) -> low residual -> N2=False
    2. Gradient orthogonal to subspace (e_7) -> high residual -> N2=True
    3. Gradient mostly in subspace (0.9*e_1 + 0.1*e_7) -> low residual -> N2=False
    4. Basis update: feed gradients in e_7 direction, verify e_7 enters basis
    5. No basis yet (first task) -> N2=False

PASS criterion: All 5 subspace tests pass with residual within 5% of expected value.
"""

import numpy as np
import pytest

from soma.core.necessity import SomaNecessity, NecessityConfig


DIM = 64  # Gradient dimensionality for tests


@pytest.fixture
def nec_with_basis() -> SomaNecessity:
    """Necessity detector with a pre-built basis from e_1, e_2, e_3."""
    cfg = NecessityConfig(subspace_rank=3, residual_threshold=0.80)
    nec = SomaNecessity(cfg)

    # Build basis from unit vectors e_1, e_2, e_3
    grads = []
    for i in range(3):
        for _ in range(20):  # 20 copies to make SVD pick them up
            e = np.zeros(DIM)
            e[i] = 1.0 + np.random.randn() * 0.01  # small perturbation
            grads.append(e)

    nec.update_basis(grads)
    return nec


class TestN2SubspaceSaturation:
    """N2: verify fires on orthogonal tasks, not on same-subspace tasks."""

    def test_gradient_in_subspace(self, nec_with_basis: SomaNecessity):
        """Gradient along e_1 (in basis) should have near-zero residual."""
        nec = nec_with_basis
        nec.reset_for_task()

        g = np.zeros(DIM)
        g[0] = 1.0
        nec.add_gradient(g)

        n2, residual = nec._check_n2()
        assert residual < 0.05, f"Residual should be <0.05 for in-subspace gradient, got {residual:.4f}"
        assert not n2, "N2 should be False for gradient already in subspace"

    def test_gradient_orthogonal(self, nec_with_basis: SomaNecessity):
        """Gradient along e_7 (orthogonal to e_1,e_2,e_3 basis) should have high residual."""
        nec = nec_with_basis
        nec.reset_for_task()

        g = np.zeros(DIM)
        g[7] = 1.0  # orthogonal to e_1, e_2, e_3
        nec.add_gradient(g)

        n2, residual = nec._check_n2()
        assert residual > 0.95, f"Residual should be >0.95 for orthogonal gradient, got {residual:.4f}"
        assert n2, "N2 should be True for orthogonal gradient"

    def test_gradient_mostly_in_subspace(self, nec_with_basis: SomaNecessity):
        """Gradient 0.9*e_1 + 0.1*e_7 should have low residual."""
        nec = nec_with_basis
        nec.reset_for_task()

        g = np.zeros(DIM)
        g[0] = 0.9
        g[7] = 0.1
        nec.add_gradient(g)

        n2, residual = nec._check_n2()
        # |0.1*e_7|^2 / |g|^2 = 0.01 / 0.82 ≈ 0.012
        assert residual < 0.05, f"Residual should be <0.05 for mostly-in-subspace gradient, got {residual:.4f}"
        assert not n2, "N2 should be False for gradient mostly in subspace"

    def test_basis_update(self):
        """After feeding many e_7 gradients and updating basis, e_7 should be in subspace."""
        # Use subspace_rank=16 so there's room for both old and new directions
        cfg = NecessityConfig(subspace_rank=16, residual_threshold=0.80)
        nec = SomaNecessity(cfg)
        grads_basis = []
        for i in range(3):
            for _ in range(20):
                e = np.zeros(DIM)
                e[i] = 1.0 + np.random.randn() * 0.01
                grads_basis.append(e)
        nec.update_basis(grads_basis)

        # Feed 50 gradients in e_7 direction
        grads_e7 = []
        for _ in range(50):
            g = np.zeros(DIM)
            g[7] = 1.0 + np.random.randn() * 0.01
            grads_e7.append(g)

        nec.update_basis(grads_e7)

        # Now e_7 should be in the basis
        nec.reset_for_task()
        g_test = np.zeros(DIM)
        g_test[7] = 1.0
        nec.add_gradient(g_test)

        n2, residual = nec._check_n2()
        assert residual < 0.05, (
            f"After basis update, e_7 residual should be <0.05, got {residual:.4f}"
        )
        assert not n2, "N2 should be False after e_7 is added to basis"

    def test_no_basis_returns_false(self):
        """With no basis (first task), N2 should be False."""
        cfg = NecessityConfig(subspace_rank=16, residual_threshold=0.80)
        nec = SomaNecessity(cfg)
        nec.reset_for_task()

        g = np.random.randn(DIM)
        nec.add_gradient(g)

        n2, residual = nec._check_n2()
        assert not n2, "N2 should be False when no basis exists"
        assert residual == 0.0, "Residual should be 0.0 when no basis exists"
