"""
Test N1 — Loss Plateau Detector.

Tests:
    1. Decreasing loss (100 steps) -> N1=False
    2. Flat loss (300+ steps) -> N1=True
    3. Noisy but improving loss -> N1=False throughout
    4. Plateau after initial drop -> N1 transitions False->True
    5. Plateau score is continuous [0,1] and monotone during plateau

PASS criterion: N1 correctly classifies 5/5 synthetic sequences.
"""

import numpy as np
import pytest

from soma.core.necessity import SomaNecessity, NecessityConfig


@pytest.fixture
def nec() -> SomaNecessity:
    """Standard necessity detector with default config."""
    cfg = NecessityConfig(
        plateau_patience=300,
        plateau_min_delta=1e-3,
        plateau_window=30,
    )
    return SomaNecessity(cfg)


class TestN1LossPlateau:
    """N1: verify fires on plateau, not on noise."""

    def test_decreasing_loss_no_plateau(self, nec: SomaNecessity):
        """Steadily decreasing loss should never trigger N1."""
        nec.reset_for_task()
        for step in range(500):
            loss = 2.0 - 0.003 * step  # steadily decreasing
            fired = nec.update_loss(loss)
            assert not fired, f"N1 should not fire on decreasing loss at step {step}"

        n1, score = nec._check_n1()
        assert not n1, "N1 should be False for steadily decreasing loss"

    def test_flat_loss_triggers_plateau(self, nec: SomaNecessity):
        """Constant loss for 300+ steps should trigger N1."""
        nec.reset_for_task()
        # First: some initial training to set best_smoothed
        for step in range(50):
            nec.update_loss(1.0 - 0.01 * step)

        # Then: flat loss for 300+ steps
        for step in range(350):
            fired = nec.update_loss(0.5)

        n1, score = nec._check_n1()
        assert n1, "N1 should be True after 300+ flat steps"
        assert score >= 1.0, f"Plateau score should be 1.0, got {score}"

    def test_noisy_improving_loss_no_plateau(self, nec: SomaNecessity):
        """Noisy but overall improving loss should not trigger N1."""
        nec.reset_for_task()
        rng = np.random.RandomState(42)
        for step in range(500):
            # Trend downward with random noise
            loss = 2.0 - 0.002 * step + rng.normal(0, 0.05)
            nec.update_loss(loss)

        n1, score = nec._check_n1()
        assert not n1, "N1 should be False for noisy but improving loss"

    def test_transition_from_improving_to_plateau(self, nec: SomaNecessity):
        """Loss that improves then plateaus should transition N1 False->True."""
        nec.reset_for_task()

        # Phase 1: improving
        for step in range(100):
            nec.update_loss(2.0 - 0.01 * step)

        n1_improving, _ = nec._check_n1()
        assert not n1_improving, "N1 should be False during improving phase"

        # Phase 2: plateau
        for step in range(350):
            nec.update_loss(1.0)

        n1_plateau, _ = nec._check_n1()
        assert n1_plateau, "N1 should be True after plateau phase"

    def test_plateau_score_monotone(self, nec: SomaNecessity):
        """Plateau score should increase monotonically during a plateau."""
        nec.reset_for_task()

        # Initial drop to set best_smoothed
        for step in range(50):
            nec.update_loss(2.0 - 0.02 * step)

        # Record scores during flat phase
        scores = []
        for step in range(350):
            nec.update_loss(1.0)
            _, score = nec._check_n1()
            scores.append(score)

        # Scores should be non-decreasing (monotone)
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 1e-9, (
                f"Plateau score not monotone: {scores[i]} < {scores[i-1]} at step {i}"
            )

        # Final score should be in [0, 1]
        assert 0.0 <= scores[-1] <= 1.0, f"Score out of range: {scores[-1]}"
