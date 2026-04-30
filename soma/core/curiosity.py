"""
SOMA Curiosity Engine
=====================
Always-on model-mismatch detector and objective generator.

Mathematical foundation:
    C(s_t) = H_epist(s_t)^2 / H_total(s_t)

Where:
    H_total  = total uncertainty (epistemic + aleatoric)
    H_epist  = learnable gap (what can be reduced by learning)
    H_aleat  = irreducible noise floor

Learnability gate:
    L(s_t) = H_epist / H_total  in [0,1]

Curiosity is NOT a reward. It is an objective-generator: it selects
what problem to solve next, not how good the last solution was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CuriosityConfig:
    """Hyperparameters for the curiosity engine."""

    # Uncertainty estimation
    n_samples: int = 8
    """Number of stochastic forward-pass samples for ensemble disagreement."""

    dropout_rate: float = 0.1
    """Dropout rate during stochastic sampling (enables MC dropout)."""

    # Learnability gate
    learnability_threshold: float = 0.3
    """Minimum L(s_t) to treat a gap as learnable. Below this = noise."""

    # Semantic entropy clustering
    n_semantic_samples: int = 10
    """Samples used for semantic entropy estimation (cluster-based)."""

    # CALM integration (Phase 2)
    use_calm: bool = False
    """If True, compute mismatch in continuous embedding space (CALM head)."""
    embedding_dim: int = 512
    """Dimensionality of CALM embedding space."""

    # Window for accumulating mismatch signals
    window_size: int = 50
    """Number of recent steps to track for rolling mismatch statistics."""

    # Noise floor estimation
    aleatoric_estimator: str = "ensemble"
    """Method for H_aleat: 'ensemble' | 'semantic' | 'dbscan_ratio'."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CuriositySignal:
    """Output of one curiosity evaluation step."""

    # Core signal
    C: float
    """Gated curiosity score in [0, 1]. High = probe this gap. Low = ignore."""

    H_total: float
    """Total uncertainty at this state."""

    H_epist: float
    """Estimated epistemic (learnable) uncertainty."""

    H_aleat: float
    """Estimated aleatoric (irreducible) uncertainty."""

    learnability: float
    """L(s_t) = H_epist / H_total."""

    # Metadata
    is_learnable: bool = False
    """True if learnability exceeds threshold — worth probing."""

    mismatch_direction: Optional[np.ndarray] = None
    """delta_t vector (CALM mode): direction of the gap in embedding space."""

    method: str = "ensemble"
    """Estimation method used."""


@dataclass
class CuriosityState:
    """Rolling state tracked across a task window."""

    mismatches: List[float] = field(default_factory=list)
    directions: List[np.ndarray] = field(default_factory=list)
    is_learnable_history: List[bool] = field(default_factory=list)
    learning_deltas: List[float] = field(default_factory=list)
    prev_epist: Optional[float] = None

    def push(self, signal: CuriositySignal, window: int) -> None:
        self.mismatches.append(signal.H_epist)
        if signal.mismatch_direction is not None:
            self.directions.append(signal.mismatch_direction)
        self.is_learnable_history.append(signal.is_learnable)
        # Rolling window
        if len(self.mismatches) > window:
            self.mismatches.pop(0)
        if len(self.directions) > window:
            self.directions.pop(0)
        if len(self.is_learnable_history) > window:
            self.is_learnable_history.pop(0)

    @property
    def mean_epist(self) -> float:
        return float(np.mean(self.mismatches)) if self.mismatches else 0.0

    @property
    def mean_learning_gain(self) -> float:
        """Average epistemic reduction observed after learning updates."""
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

    def record_learning_outcome(self, pre_epist: float, post_epist: float, window: int) -> float:
        """Close the loop by recording whether learning reduced epistemic uncertainty."""
        delta = pre_epist - post_epist
        self.learning_deltas.append(delta)
        if len(self.learning_deltas) > window:
            self.learning_deltas.pop(0)
        self.prev_epist = post_epist
        return delta


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class CuriosityEngine:
    """
    Always-on mismatch detector and learning-objective generator.

    Usage
    -----
    engine = CuriosityEngine(cfg)

    # At each inference step:
    signal = engine.evaluate(
        logits_samples=samples,   # list of logit arrays (ensemble)
        true_next=y_true,         # ground truth next token / embedding
    )

    # Feed result to retrieval, verification, and necessity check:
    if signal.is_learnable:
        retrieve_and_clarify()
        verify()
        self_learn()
        if necessity_triggered():
            grow()
        else:
            answer()
    """

    def __init__(self, cfg: Optional[CuriosityConfig] = None) -> None:
        self.cfg = cfg or CuriosityConfig()
        self.state = CuriosityState()
        self._calibrated_noise_floor: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        logits_samples: List[np.ndarray],
        true_next: Optional[np.ndarray] = None,
        embeddings_samples: Optional[List[np.ndarray]] = None,
    ) -> CuriositySignal:
        """
        Evaluate curiosity at the current state.

        Parameters
        ----------
        logits_samples : list of [vocab_size] arrays
            K stochastic samples from the model head (MC dropout or ensemble).
        true_next : array of shape [vocab_size] or [embedding_dim]
            Ground truth next token (one-hot) or next embedding (CALM mode).
        embeddings_samples : list of [embedding_dim] arrays (CALM mode)
            K stochastic embedding predictions from f_phi(h_t).

        Returns
        -------
        CuriositySignal
        """
        if self.cfg.use_calm and embeddings_samples is not None:
            signal = self._evaluate_calm(embeddings_samples, true_next)
        else:
            signal = self._evaluate_softmax(logits_samples)

        self.state.push(signal, self.cfg.window_size)
        return signal

    def window_summary(self) -> dict:
        """Rolling statistics over the current task window."""
        return {
            "mean_H_epist": self.state.mean_epist,
            "mean_learning_gain": self.state.mean_learning_gain,
            "learnable_fraction": self.state.learnable_fraction,
            "window_len": len(self.state.mismatches),
        }

    def record_learning_outcome(self, pre_signal: CuriositySignal, post_signal: CuriositySignal) -> float:
        """Record epistemic uncertainty drop after a learning attempt."""
        return self.state.record_learning_outcome(
            pre_signal.H_epist,
            post_signal.H_epist,
            self.cfg.window_size,
        )

    def reset(self) -> None:
        """Clear rolling state between tasks."""
        self.state = CuriosityState()

    def calibrate_noise_floor(self, baseline_samples: List[float]) -> None:
        """
        Set the aleatoric noise floor from baseline observations
        (e.g., from cold-start tasks where behaviour is understood).
        """
        self._calibrated_noise_floor = float(np.mean(baseline_samples))

    # ------------------------------------------------------------------
    # Softmax / discrete mode
    # ------------------------------------------------------------------

    def _evaluate_softmax(self, logits_samples: List[np.ndarray]) -> CuriositySignal:
        """
        Ensemble-disagreement estimator in discrete token space.

        H_epist = H[mean(P_k)] - mean(H[P_k])
                = mutual information between model index and prediction.

        This is the standard method; its limitation is that it conflates
        semantically-similar-but-lexically-different tokens.
        See CALM mode for the richer geometric estimator.
        """
        probs = [self._softmax(l) for l in logits_samples]
        P_mean = np.mean(probs, axis=0)          # mixture distribution

        H_total = self._entropy(P_mean)           # H[mean(P_k)]
        H_aleat = float(np.mean([self._entropy(p) for p in probs]))  # mean H[P_k]
        H_epist = max(H_total - H_aleat, 0.0)

        return self._build_signal(H_total, H_epist, H_aleat, method="ensemble_softmax")

    # ------------------------------------------------------------------
    # CALM / continuous embedding mode
    # ------------------------------------------------------------------

    def _evaluate_calm(
        self,
        embedding_samples: List[np.ndarray],
        true_embedding: Optional[np.ndarray],
    ) -> CuriositySignal:
        """
        Geometric mismatch estimator in continuous embedding space.

        delta_t^(k) = f_phi^(k)(h_t) - v*_{t+1}

        H_epist ≈ ||mean(delta_t)||^2   (systematic error: learnable)
        H_aleat ≈ var(delta_t)          (noise around mean: irreducible)
        C(s_t)  = H_epist^2 / H_total

        Advantages over softmax:
        - Geometry is semantic (distance = meaning distance)
        - N3 clustering is in d-dim space, not 100K logit space
        - Cross-lingual tasks land in same space (no vocabulary wall)
        """
        stacked = np.stack(embedding_samples)     # [K, d]
        delta_mean = np.mean(stacked, axis=0)     # systematic mismatch direction

        if true_embedding is not None:
            delta_mean = delta_mean - true_embedding

        H_epist = float(np.dot(delta_mean, delta_mean))       # ||mean delta||^2
        H_aleat_vec = np.var(stacked, axis=0)                 # per-dim variance
        H_aleat = float(np.sum(H_aleat_vec))                  # total noise floor
        H_total = H_epist + H_aleat

        return self._build_signal(
            H_total, H_epist, H_aleat,
            method="calm_geometric",
            direction=delta_mean,
        )

    # ------------------------------------------------------------------
    # Signal assembly and gating
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        H_total: float,
        H_epist: float,
        H_aleat: float,
        method: str,
        direction: Optional[np.ndarray] = None,
    ) -> CuriositySignal:
        if self._calibrated_noise_floor is not None:
            H_aleat = max(H_aleat, self._calibrated_noise_floor)
            H_total = H_epist + H_aleat

        eps = 1e-9
        
        # -------------------------------
        # 1. Learnability
        # -------------------------------
        learnability = H_epist / (H_total + eps)

        # -------------------------------
        # 2. Generalization proxy (G)
        # Use rolling epistemic average and observed learning gains as a cheap proxy
        # -------------------------------
        G = 1.0 + self.state.mean_epist + max(self.state.mean_learning_gain, 0.0)

        # -------------------------------
        # 3. Novelty term (N)
        # Penalize repeated exposure
        # -------------------------------
        visit_count = len(self.state.mismatches)
        beta = 0.05
        N = math.exp(-beta * visit_count)

        # -------------------------------
        # 4. Final curiosity
        # -------------------------------
        gamma = 2.0

        C = H_epist * (learnability ** gamma) * G * N

        improvement = self.state.improvement(H_epist)
        if improvement > 0:
            C *= (1.0 + improvement)

        # Stable normalization
        C_norm = C / (1.0 + C)

        return CuriositySignal(
            C=C_norm,
            H_total=H_total,
            H_epist=H_epist,
            H_aleat=H_aleat,
            learnability=learnability,
            is_learnable=(learnability >= self.cfg.learnability_threshold),
            mismatch_direction=direction,
            method=method,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        x = logits - np.max(logits)
        e = np.exp(x)
        return e / e.sum()

    @staticmethod
    def _entropy(probs: np.ndarray) -> float:
        eps = 1e-12
        return float(-np.sum(probs * np.log(probs + eps)))


# ---------------------------------------------------------------------------
# Retrieve & Clarify stub (wired to curiosity output)
# ---------------------------------------------------------------------------

class RetrieveAndClarify:
    """
    Step 2 in the SOMA learning loop.

    Triggered when CuriosityEngine detects a learnable mismatch.
    Always searches — even for things that might be known — to ensure
    world-knowledge alignment before the Verify step.

    In Paper 1: stub (returns None, gap filled by human curriculum).
    In Paper 2: wired to real web search + clarification Q-generation.
    """

    def __init__(self, always_search: bool = True) -> None:
        self.always_search = always_search

    def run(self, signal: CuriositySignal, context: str = "") -> dict:
        """
        Given a curiosity signal, retrieve relevant world knowledge.

        Returns a dict with:
          'retrieved_docs': list of doc strings (stub: empty)
          'clarifying_questions': list of questions (stub: empty)
          'signal': the input CuriositySignal
        """
        if not signal.is_learnable:
            return {
                "retrieved_docs": [],
                "clarifying_questions": [],
                "signal": signal,
                "skipped": True,
            }

        # Paper 1 stub — replace with actual search in Paper 2
        retrieved = self._search(context)
        questions = self._generate_questions(signal, context)

        return {
            "retrieved_docs": retrieved,
            "clarifying_questions": questions,
            "signal": signal,
            "skipped": False,
        }

    def _search(self, context: str) -> list:
        """Stub: replace with Qwen3.6-27B tool-use search."""
        return []  # TODO Paper 2: integrate web_search tool

    def _generate_questions(self, signal: CuriositySignal, context: str) -> list:
        """Stub: generate clarifying questions from the mismatch direction."""
        return []  # TODO Paper 2: use direction vector to identify gap type


# ---------------------------------------------------------------------------
# Verify stub (Lean / Z3 integration point)
# ---------------------------------------------------------------------------

class Verifier:
    """
    Step 3 in the SOMA learning loop.

    Checks:
    (a) Internal consistency — uses Z3 SMT solver for logical claims
    (b) World-knowledge alignment — checks against retrieved docs
    (c) Formal proof validity — Lean 4 for mathematical assertions

    Paper 1: stub. Paper 3: full integration with Lean 4 + Z3.
    """

    def verify(
        self,
        statement: str,
        retrieved_docs: list,
        use_z3: bool = False,
        use_lean: bool = False,
    ) -> dict:
        """
        Returns:
          'consistent': bool
          'confidence': float
          'method': str
        """
        # Stub — always returns consistent for Paper 1
        return {
            "consistent": True,
            "confidence": 1.0,
            "method": "stub_paper1",
        }
        # TODO Paper 3:
        # if use_z3:
        #     return self._z3_check(statement)
        # if use_lean:
        #     return self._lean4_check(statement)
