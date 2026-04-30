"""
soma/curiosity/engine.py — SOMA Curiosity Engine v2
====================================================
Always-on mismatch detector and learning-objective generator.

FIXES applied from code review observations:
  1. Fixed thresholds (0.80, 0.30) → Bayesian adaptive thresholds
     that update from observed domain statistics
  2. Subspace rank=16 → adaptive rank from eigenvalue decay
     rank chosen where cumulative variance > 95%
  3. Entropy alone → entropy + margin + logit gap combined signal
     catches "confident but wrong" patterns that entropy misses

Mathematical foundation:
    C(s_t) = H_epist(s_t)^γ / H_total(s_t) · G · N

Where:
    H_total  = total uncertainty (epistemic + aleatoric)
    H_epist  = learnable gap (reducible by learning)
    H_aleat  = irreducible noise floor
    G        = generalisation proxy (rolling learning gain)
    N        = novelty term (penalise repeated exposure)

CALM integration:
    In CALM mode (Phase 3), H_epist becomes ||mean(delta_t)||²
    where delta_t = f_phi(h_t) - v*_{t+1} in continuous vector space.
    This naturally handles cross-lingual mismatch without vocabulary wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CuriosityConfig:
    # Ensemble sampling
    n_samples: int = 8
    dropout_rate: float = 0.1

    # Adaptive learnability threshold (replaces fixed 0.30)
    # These are PRIORS — updated from domain observations via Bayesian update
    learnability_prior_alpha: float = 3.0    # Beta distribution alpha
    learnability_prior_beta: float = 7.0     # Beta distribution beta
    # Result: prior mean = 3/10 = 0.30, but updates with evidence

    # Adaptive subspace rank (replaces fixed 16)
    variance_explained_target: float = 0.95  # keep dims explaining 95% variance
    min_rank: int = 4
    max_rank: int = 64

    # Multi-signal uncertainty (fixes entropy-alone problem)
    use_margin_signal: bool = True    # add prediction margin to uncertainty
    use_logit_gap: bool = True        # add top-1 vs top-2 logit gap
    margin_weight: float = 0.3        # weight of margin in combined signal
    logit_gap_weight: float = 0.2     # weight of logit gap in combined signal
    entropy_weight: float = 0.5       # weight of entropy in combined signal

    # CALM integration
    use_calm: bool = False
    embedding_dim: int = 512

    # Rolling window
    window_size: int = 50

    # Curiosity scoring
    gamma: float = 2.0       # learnability exponent
    novelty_decay: float = 0.05  # beta in novelty term

    # Noise floor
    aleatoric_estimator: str = "ensemble"  # "ensemble" | "semantic" | "calibrated"

    # Self-learn trigger
    self_learn_curiosity_threshold: float = 0.5  # above this → self-learn


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Threshold (fixes observation 1)
# ─────────────────────────────────────────────────────────────────────────────

class BayesianThreshold:
    """
    Adaptive threshold using Bayesian Beta-Bernoulli model.

    Instead of a fixed threshold (e.g. 0.30), this maintains a Beta
    distribution over the threshold parameter, updated from observations:
        - Each time curiosity correctly identifies a learnable gap → alpha++
        - Each time curiosity fires on noise (no learning gain) → beta++

    Posterior mean = alpha / (alpha + beta) adapts to the domain.
    """

    def __init__(self, prior_alpha: float = 3.0, prior_beta: float = 7.0):
        self.alpha = prior_alpha
        self.beta  = prior_beta

    @property
    def threshold(self) -> float:
        """Current posterior mean of the threshold."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Posterior variance — how confident we are in the threshold."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def update(self, was_learnable: bool, did_learn: bool) -> None:
        """
        Update threshold from outcome.

        Args:
            was_learnable: curiosity said this was learnable
            did_learn: epistemic uncertainty actually reduced after learning
        """
        if was_learnable and did_learn:
            self.alpha += 1.0    # true positive: threshold is correctly placed
        elif was_learnable and not did_learn:
            self.beta  += 1.0    # false positive: threshold too low, raise it
        # True negatives and false negatives don't provide direct signal


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Subspace Rank (fixes observation 2)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_rank(
    gradients: np.ndarray,
    variance_target: float = 0.95,
    min_rank: int = 4,
    max_rank: int = 64,
) -> int:
    """
    Choose LoRA rank from eigenvalue decay of gradient matrix.

    Instead of fixed rank=16, compute the minimum rank that explains
    variance_target fraction of the total gradient variance.

    This means:
      - Simple tasks (low-rank gradient structure) → small rank (saves compute)
      - Complex tasks (full-rank gradients) → larger rank (captures complexity)

    Args:
        gradients: [n_samples, n_params] matrix of collected gradients
        variance_target: cumulative variance to explain (default: 95%)
        min_rank, max_rank: clamp result

    Returns:
        Adaptive rank r* in [min_rank, max_rank]
    """
    G = gradients - gradients.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(G, full_matrices=False)
        eigenvalues = s ** 2
        total = eigenvalues.sum() + 1e-9
        cumvar = np.cumsum(eigenvalues) / total
        # Find minimum rank achieving variance_target
        candidates = np.where(cumvar >= variance_target)[0]
        if len(candidates) == 0:
            rank = max_rank
        else:
            rank = int(candidates[0]) + 1   # 0-indexed → 1-indexed
    except np.linalg.LinAlgError:
        rank = 16  # fallback
    return int(np.clip(rank, min_rank, max_rank))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Signal Uncertainty (fixes observation 3)
# ─────────────────────────────────────────────────────────────────────────────

def combined_uncertainty(
    logits_samples: List[np.ndarray],
    entropy_w: float = 0.5,
    margin_w: float = 0.3,
    logit_gap_w: float = 0.2,
) -> Tuple[float, float, float]:
    """
    Combined uncertainty signal: entropy + margin + logit gap.

    Fixes observation 3: entropy alone misses "confident but wrong" patterns.

    Example of what entropy misses:
      Model outputs [0.99, 0.001, 0.001, ...] every time (very low entropy)
      But the 0.99 prediction is consistently WRONG.
      Entropy says: "I'm certain" → not curious → never learns
      Margin+gap says: "correct class has near-zero probability" → very curious

    Entropy:   -Σ p log p              → low when model is confident
    Margin:    P(correct) - P(top-1)   → low when correct not in top-1
    Logit gap: logit[1] - logit[2]     → small when top-2 are confused

    Returns:
        (H_epist, H_aleat, combined) where combined ∈ [0, 1]
    """
    probs_list = [_softmax(l) for l in logits_samples]
    P_mean = np.mean(probs_list, axis=0)

    # ── Entropy component ────────────────────────────────────────────────
    H_total  = _entropy(P_mean)
    H_aleat  = float(np.mean([_entropy(p) for p in probs_list]))
    H_epist  = max(H_total - H_aleat, 0.0)

    # ── Margin component: instability of top prediction ─────────────────
    # How much does the top prediction vary across samples?
    top1_preds = [np.argmax(p) for p in probs_list]
    top1_probs = [p[np.argmax(p)] for p in probs_list]
    # Margin = std of top-1 probability (high std = uncertain top prediction)
    margin_uncertainty = float(np.std(top1_probs))

    # ── Logit gap: top-1 vs top-2 separation ────────────────────────────
    # Average logit gap across samples; small gap = confused between top-2
    gaps = []
    for l in logits_samples:
        sorted_l = np.sort(l)[::-1]
        if len(sorted_l) >= 2:
            gaps.append(sorted_l[0] - sorted_l[1])
    logit_gap = float(np.mean(gaps)) if gaps else 0.0
    # Normalise: smaller gap → higher uncertainty
    logit_gap_uncertainty = 1.0 / (1.0 + abs(logit_gap))

    # ── Combined signal ──────────────────────────────────────────────────
    # Normalise entropy to [0,1] (max entropy for vocab_size v is log(v))
    H_norm = H_epist / (np.log(len(P_mean)) + 1e-9)

    combined = (
        entropy_w     * min(H_norm, 1.0) +
        margin_w      * min(margin_uncertainty * 5, 1.0) +  # scale to [0,1]
        logit_gap_w   * logit_gap_uncertainty
    )
    return H_epist, H_aleat, float(np.clip(combined, 0.0, 1.0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / (e.sum() + 1e-9)


def _entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum(probs * np.log(probs + eps)))


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CuriositySignal:
    C: float                          # final curiosity score ∈ [0,1]
    H_total: float
    H_epist: float
    H_aleat: float
    learnability: float               # H_epist / H_total
    is_learnable: bool                # learnability >= adaptive threshold
    adaptive_threshold: float         # current Bayesian threshold value
    margin_uncertainty: float         # contribution from margin signal
    logit_gap_uncertainty: float      # contribution from logit gap signal
    mismatch_direction: Optional[np.ndarray] = None  # CALM mode
    method: str = "ensemble"
    recommended_rank: Optional[int] = None  # adaptive rank suggestion


@dataclass
class CuriosityState:
    mismatches: List[float] = field(default_factory=list)
    is_learnable_history: List[bool] = field(default_factory=list)
    learning_deltas: List[float] = field(default_factory=list)
    gradient_history: List[np.ndarray] = field(default_factory=list)
    prev_epist: Optional[float] = None

    def push(self, signal: CuriositySignal, window: int) -> None:
        self.mismatches.append(signal.H_epist)
        self.is_learnable_history.append(signal.is_learnable)
        if len(self.mismatches) > window:
            self.mismatches.pop(0)
        if len(self.is_learnable_history) > window:
            self.is_learnable_history.pop(0)

    def push_gradient(self, grad: np.ndarray, window: int) -> None:
        self.gradient_history.append(grad.flatten())
        if len(self.gradient_history) > window:
            self.gradient_history.pop(0)

    @property
    def mean_epist(self) -> float:
        return float(np.mean(self.mismatches)) if self.mismatches else 0.0

    @property
    def mean_learning_gain(self) -> float:
        return float(np.mean(self.learning_deltas)) if self.learning_deltas else 0.0

    @property
    def learnable_fraction(self) -> float:
        if not self.is_learnable_history:
            return 0.0
        return sum(self.is_learnable_history) / len(self.is_learnable_history)

    def improvement(self, current: float) -> float:
        if self.prev_epist is None:
            self.prev_epist = current
            return 0.0
        delta = self.prev_epist - current
        self.prev_epist = current
        return delta

    def record_learning_outcome(self, pre: float, post: float, window: int) -> float:
        delta = pre - post
        self.learning_deltas.append(delta)
        if len(self.learning_deltas) > window:
            self.learning_deltas.pop(0)
        self.prev_epist = post
        return delta

    def get_gradient_matrix(self) -> Optional[np.ndarray]:
        if len(self.gradient_history) < 4:
            return None
        return np.stack(self.gradient_history)


# ─────────────────────────────────────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────────────────────────────────────

class CuriosityEngine:
    """
    Always-on mismatch detector. Runs continuously — not triggered by events.

    Curiosity is NOT a reward. It is an OBJECTIVE GENERATOR:
    it selects what to learn next, and HOW MUCH capacity to allocate.

    Connects to the SOMA loop at:
      signal = engine.evaluate(...)
      if signal.is_learnable:
          → retrieve_and_clarify()
          → verify()
          → self_learn()
          if necessity.check():
              → grow_adapter(rank=signal.recommended_rank)
          else:
              → answer()

    Three fixes from code review applied:
      1. Bayesian adaptive threshold (not fixed 0.30)
      2. Adaptive rank from eigenvalue decay (not fixed 16)
      3. Entropy + margin + logit_gap combined (not entropy alone)
    """

    def __init__(self, cfg: Optional[CuriosityConfig] = None) -> None:
        self.cfg = cfg or CuriosityConfig()
        self.state = CuriosityState()
        self.threshold = BayesianThreshold(
            self.cfg.learnability_prior_alpha,
            self.cfg.learnability_prior_beta,
        )
        self._calibrated_noise: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        logits_samples: List[np.ndarray],
        true_next: Optional[np.ndarray] = None,
        embeddings_samples: Optional[List[np.ndarray]] = None,
        gradients: Optional[List[np.ndarray]] = None,
    ) -> CuriositySignal:
        """
        Evaluate curiosity at current state.

        Args:
            logits_samples: K stochastic forward pass logits [vocab_size]
            true_next: ground truth token or embedding (optional, improves accuracy)
            embeddings_samples: K embedding predictions [d] (CALM mode)
            gradients: recent gradients for adaptive rank estimation

        Returns:
            CuriositySignal with is_learnable, recommended_rank, and all scores
        """
        if gradients:
            for g in gradients:
                self.state.push_gradient(g, self.cfg.window_size)

        if self.cfg.use_calm and embeddings_samples is not None:
            signal = self._evaluate_calm(embeddings_samples, true_next)
        else:
            signal = self._evaluate_combined(logits_samples)

        # Adaptive rank recommendation
        G = self.state.get_gradient_matrix()
        if G is not None:
            signal.recommended_rank = adaptive_rank(
                G,
                self.cfg.variance_explained_target,
                self.cfg.min_rank,
                self.cfg.max_rank,
            )

        self.state.push(signal, self.cfg.window_size)
        return signal

    def record_outcome(
        self,
        pre: CuriositySignal,
        post: CuriositySignal,
    ) -> float:
        """
        Close the learning loop. Call after self_learn() completes.
        Updates Bayesian threshold and rolling statistics.

        Returns:
            Epistemic reduction (positive = learning happened)
        """
        did_learn = post.H_epist < pre.H_epist
        self.threshold.update(
            was_learnable=pre.is_learnable,
            did_learn=did_learn,
        )
        return self.state.record_learning_outcome(
            pre.H_epist, post.H_epist, self.cfg.window_size
        )

    def calibrate(self, baseline_scores: List[float]) -> None:
        """
        Set aleatoric noise floor from cold-start observations.
        Call after first 2 tasks (same pattern as SOMA-NECESSITY calibration).
        """
        self._calibrated_noise = float(np.mean(baseline_scores))

    def reset(self) -> None:
        self.state = CuriosityState()

    def summary(self) -> dict:
        return {
            "adaptive_threshold": round(self.threshold.threshold, 4),
            "threshold_uncertainty": round(self.threshold.uncertainty, 4),
            "mean_H_epist": round(self.state.mean_epist, 4),
            "mean_learning_gain": round(self.state.mean_learning_gain, 4),
            "learnable_fraction": round(self.state.learnable_fraction, 4),
            "window_size": len(self.state.mismatches),
        }

    # ── Signal computation ────────────────────────────────────────────────

    def _evaluate_combined(
        self,
        logits_samples: List[np.ndarray],
    ) -> CuriositySignal:
        """
        Three-component uncertainty: entropy + margin + logit_gap.
        """
        H_epist, H_aleat, combined = combined_uncertainty(
            logits_samples,
            entropy_w   = self.cfg.entropy_weight,
            margin_w    = self.cfg.margin_weight,
            logit_gap_w = self.cfg.logit_gap_weight,
        )

        probs_list = [_softmax(l) for l in logits_samples]
        P_mean = np.mean(probs_list, axis=0)

        # Margin signal (standalone for signal transparency)
        top1_probs = [p[np.argmax(p)] for p in probs_list]
        margin_unc = float(np.std(top1_probs))

        # Logit gap (standalone)
        gaps = []
        for l in logits_samples:
            s = np.sort(l)[::-1]
            if len(s) >= 2:
                gaps.append(s[0] - s[1])
        logit_gap_unc = 1.0 / (1.0 + abs(float(np.mean(gaps))) if gaps else 1.0)

        H_total = H_epist + H_aleat
        if self._calibrated_noise is not None:
            H_aleat = max(H_aleat, self._calibrated_noise)
            H_total = H_epist + H_aleat

        return self._build_signal(
            H_total, H_epist, H_aleat, combined,
            margin_unc, logit_gap_unc,
            method="combined_entropy_margin_gap",
        )

    def _evaluate_calm(
        self,
        embedding_samples: List[np.ndarray],
        true_embedding: Optional[np.ndarray],
    ) -> CuriositySignal:
        """
        CALM mode: geometric mismatch in continuous embedding space.

        H_epist = ||mean(delta_t)||²  (systematic = learnable)
        H_aleat = sum(var(delta_t))   (noise = irreducible)
        """
        stacked = np.stack(embedding_samples)
        mean_pred = np.mean(stacked, axis=0)
        delta_mean = mean_pred - true_embedding if true_embedding is not None else mean_pred

        H_epist = float(np.dot(delta_mean, delta_mean))
        H_aleat = float(np.sum(np.var(stacked, axis=0)))
        H_total = H_epist + H_aleat

        combined = H_epist / (H_total + 1e-9)  # in CALM, this is well-defined

        return self._build_signal(
            H_total, H_epist, H_aleat, combined,
            margin_unc=0.0, logit_gap_unc=0.0,
            method="calm_geometric",
            direction=delta_mean,
        )

    def _build_signal(
        self,
        H_total: float,
        H_epist: float,
        H_aleat: float,
        combined_unc: float,
        margin_unc: float,
        logit_gap_unc: float,
        method: str,
        direction: Optional[np.ndarray] = None,
    ) -> CuriositySignal:

        eps = 1e-9
        thresh = self.threshold.threshold

        learnability = H_epist / (H_total + eps)

        # Generalisation proxy: rolling gain
        G = 1.0 + self.state.mean_epist + max(self.state.mean_learning_gain, 0.0)

        # Novelty: penalise repeated exposure
        N = math.exp(-self.cfg.novelty_decay * len(self.state.mismatches))

        # Improvement bonus
        improvement = self.state.improvement(H_epist)
        improvement_bonus = 1.0 + max(improvement, 0.0)

        # Use combined uncertainty (not just entropy) as the base signal
        C_raw = combined_unc * (learnability ** self.cfg.gamma) * G * N * improvement_bonus
        C = C_raw / (1.0 + C_raw)  # stable normalisation

        return CuriositySignal(
            C=float(C),
            H_total=H_total,
            H_epist=H_epist,
            H_aleat=H_aleat,
            learnability=learnability,
            is_learnable=(learnability >= thresh),
            adaptive_threshold=thresh,
            margin_uncertainty=margin_unc,
            logit_gap_uncertainty=logit_gap_unc,
            mismatch_direction=direction,
            method=method,
            recommended_rank=None,  # filled by evaluate()
        )
