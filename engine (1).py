"""
soma/necessity/engine.py — SOMA-NECESSITY v2
============================================
Updated to connect cleanly with the Curiosity Engine.

Key changes from v1:
  1. Necessity now receives CuriositySignal as additional input
     — curiosity pre-screens, necessity makes the final call
  2. Adaptive subspace rank (from curiosity's recommended_rank)
     instead of fixed 16
  3. N3 entropy threshold uses same Bayesian calibration pattern
     as CuriosityEngine.threshold — consistent across the system

Flow in the full loop:
  curiosity.evaluate() → is_learnable=True → retrieve → verify → self_learn
  if confidence still low after self_learn:
    necessity.check() → N1∧N2∧N3 → grow or not

Necessity is NOT curiosity. They answer different questions:
  Curiosity: "Is there something worth learning here?"
  Necessity: "Has the existing capacity truly been exhausted?"

Curiosity fires frequently (it's always-on).
Necessity fires rarely (only when self-learning failed).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class NecessityConfig:
    # N1 — loss plateau
    plateau_patience: int    = 300
    plateau_min_delta: float = 1e-3
    plateau_window: int      = 30

    # N2 — subspace (adaptive rank passed in from curiosity)
    residual_threshold: float = 0.80  # still fixed but can be overridden
    # If curiosity provides recommended_rank, use that; else use this default
    default_rank: int         = 16

    # N3 — systematic failure clustering
    min_failures: int         = 25
    proj_dim: int             = 32
    silhouette_min: float     = 0.30
    entropy_default: float    = 0.70

    # Calibration — mirrors CuriosityEngine pattern
    calibration_tasks: int    = 2


@dataclass
class NecessityResult:
    necessity: bool
    n1: bool
    n2: bool
    n3: bool
    plateau_score: float
    residual_fraction: float
    silhouette: float
    entropy: float
    rank_used: int


class LossPlateauDetector:
    def __init__(self, cfg: NecessityConfig):
        self.cfg = cfg
        self._history: list[float] = []
        self._best = float("inf")
        self._stale = 0

    def update(self, loss: float) -> bool:
        self._history.append(loss)
        if len(self._history) < self.cfg.plateau_window:
            return False
        s = float(np.mean(self._history[-self.cfg.plateau_window:]))
        if (self._best - s) > self.cfg.plateau_min_delta:
            self._best = s; self._stale = 0
        else:
            self._stale += 1
        return self._stale >= self.cfg.plateau_patience

    @property
    def score(self) -> float:
        return min(1.0, self._stale / max(1, self.cfg.plateau_patience))

    def reset(self):
        self._history.clear(); self._best = float("inf"); self._stale = 0


class SubspaceDetector:
    def __init__(self, cfg: NecessityConfig):
        self.cfg = cfg
        self._basis: Optional[np.ndarray] = None
        self._buf: list[np.ndarray] = []
        self._last_residual = 0.5

    def update_basis(self, grads: list[np.ndarray], rank: Optional[int] = None):
        if not grads:
            return
        r = rank or self.cfg.default_rank
        G = np.stack([g.flatten() for g in grads])
        G -= G.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(G, full_matrices=False)
            new_vecs = Vt[:r].T
            if self._basis is None:
                self._basis = new_vecs
            else:
                combined = np.concatenate([self._basis, new_vecs], axis=1)
                Q, _ = np.linalg.qr(combined)
                self._basis = Q[:, :r]
        except np.linalg.LinAlgError:
            pass

    def add_grad(self, g: np.ndarray):
        self._buf.append(g.flatten())

    def check(self) -> tuple[bool, float]:
        if self._basis is None or len(self._buf) < 5:
            self._buf.clear()
            return False, 0.5
        g = np.stack(self._buf).mean(0)
        g = g / (np.linalg.norm(g) + 1e-8)
        proj = self._basis @ (self._basis.T @ g)
        res = float(np.dot(g - proj, g - proj) / (np.dot(g, g) + 1e-8))
        self._last_residual = res
        self._buf.clear()
        return res > self.cfg.residual_threshold, res


class SystematicFailureDetector:
    def __init__(self, cfg: NecessityConfig):
        self.cfg = cfg
        self._grads: list[np.ndarray] = []
        self._cal_entropies: list[float] = []
        self._threshold: Optional[float] = None

    def add(self, g: np.ndarray):
        self._grads.append(g.flatten())

    def calibrate(self, e: float):
        self._cal_entropies.append(e)
        if len(self._cal_entropies) >= self.cfg.calibration_tasks:
            self._threshold = 0.70 * float(np.mean(self._cal_entropies))

    def _measure(self, grads: list[np.ndarray]) -> tuple[float, float]:
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics import silhouette_score
        except ImportError:
            return 0.0, 1.0
        G = np.stack(grads) - np.mean(grads, axis=0)
        try:
            _, _, Vt = np.linalg.svd(G, full_matrices=False)
            d = min(self.cfg.proj_dim, Vt.shape[0])
            G = G @ Vt[:d].T
        except np.linalg.LinAlgError:
            return 0.0, 1.0
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        G /= (norms + 1e-8)
        labels = DBSCAN(eps=0.5, min_samples=5, metric="cosine").fit_predict(G)
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k < 2:
            return 0.0, 1.0
        mask = labels != -1
        if mask.sum() < 4:
            return 0.0, 1.0
        sil = float(silhouette_score(G[mask], labels[mask], metric="cosine"))
        p = np.bincount(labels[mask]) / mask.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        return sil, entropy

    def get_entropy(self) -> float:
        if len(self._grads) < self.cfg.min_failures:
            return self.cfg.entropy_default
        _, e = self._measure(self._grads)
        return e

    def check(self) -> tuple[bool, float, float]:
        if len(self._grads) < self.cfg.min_failures:
            return False, 0.0, 1.0
        sil, entropy = self._measure(self._grads)
        thresh = self._threshold if self._threshold is not None else self.cfg.entropy_default
        n3 = (sil > self.cfg.silhouette_min) and (entropy < thresh)
        self._grads.clear()
        return n3, sil, entropy


class SomaNecessity:
    """
    SOMA-NECESSITY v2 — integrates with CuriosityEngine.

    Receives curiosity's recommended_rank for adaptive subspace.
    Only called when self-learning was insufficient.

    Usage in the full loop:
        # After self_learn():
        if confidence < threshold:
            nec = necessity.check(recommended_rank=signal.recommended_rank)
            if nec.necessity:
                grow_adapter(rank=nec.rank_used)
    """

    def __init__(self, cfg: Optional[NecessityConfig] = None):
        self.cfg = cfg or NecessityConfig()
        self.n1 = LossPlateauDetector(self.cfg)
        self.n2 = SubspaceDetector(self.cfg)
        self.n3 = SystematicFailureDetector(self.cfg)
        self._cal_done = False
        self._cal_count = 0

    def update_loss(self, loss: float): return self.n1.update(loss)
    def add_gradient(self, g: np.ndarray): self.n2.add_grad(g)
    def add_failure_gradient(self, g: np.ndarray): self.n3.add(g)

    def task_completed(self, grads: list[np.ndarray], rank: Optional[int] = None):
        """
        Call after task finishes. Passes adaptive rank to subspace detector.
        """
        self.n2.update_basis(grads, rank=rank)
        if not self._cal_done:
            self.n3.calibrate(self.n3.get_entropy())
            self._cal_count += 1
            if self._cal_count >= self.cfg.calibration_tasks:
                self._cal_done = True

    def reset_for_task(self):
        self.n1.reset()
        self.n2._buf.clear()
        self.n3._grads.clear()

    def check(self, recommended_rank: Optional[int] = None) -> NecessityResult:
        """
        Run full N1∧N2∧N3 check.

        Args:
            recommended_rank: from CuriosityEngine.evaluate() — adaptive rank.
                              If provided, used for subspace computation.
        """
        rank_used = recommended_rank or self.cfg.default_rank

        n1 = self.n1.score >= 1.0
        n2, res = self.n2.check()
        n3, sil, ent = self.n3.check()

        return NecessityResult(
            necessity=n1 and n2 and n3,
            n1=n1, n2=n2, n3=n3,
            plateau_score=self.n1.score,
            residual_fraction=res,
            silhouette=sil,
            entropy=ent,
            rank_used=rank_used,
        )

    def rl_state_features(self) -> np.ndarray:
        """4 features for RL state vector."""
        return np.array([
            self.n1.score,
            self.n2._last_residual,
            self.n3.get_entropy(),
            min(1.0, len(self.n3._grads) / max(1, self.cfg.min_failures)),
        ], dtype=np.float32)
