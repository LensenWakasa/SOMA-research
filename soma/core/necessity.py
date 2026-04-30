"""
SOMA-NECESSITY — Three-signal capacity trigger.

Implements the jointly-necessary condition detector:
    N1 — Loss Plateau: training has genuinely stalled
    N2 — Subspace Saturation: new gradient is orthogonal to existing principal subspace
    N3 — Systematic Failure: failure-case gradients cluster consistently

Growth is triggered ONLY when all three signals fire simultaneously.
This prevents the primary failure mode of prior methods: over-spawning on false positives.

References:
    - KeepLoRA (Jan 2026) — subspace saturation methodology
    - InfLoRA (CVPR 2024) — gradient subspace analysis
    - TreeLoRA (ICML 2025) — gradient similarity for adapter organisation
    - SplitLoRA — residual threshold 0.80 analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NecessityConfig:
    """Configuration for the SOMA-NECESSITY detector.

    All defaults match Section 6.1 of the engineering specification.
    """

    # N1 — Loss Plateau
    plateau_patience: int = 300
    """Steps without improvement before N1=True."""
    plateau_min_delta: float = 1e-3
    """Minimum meaningful loss improvement."""
    plateau_window: int = 30
    """Rolling window size for loss smoothing."""

    # N2 — Subspace Saturation
    subspace_rank: int = 16
    """Number of principal SVD directions to keep."""
    residual_threshold: float = 0.80
    """Fraction of gradient energy in residual subspace to declare saturation."""

    # N3 — Systematic Failure Detection
    min_failures: int = 25
    """Minimum failure cases needed before N3 clustering."""
    proj_dim: int = 32
    """Gradient projection dimension for DBSCAN."""
    silhouette_min: float = 0.30
    """Minimum silhouette score for real clusters."""
    entropy_default: float = 0.70
    """Fallback entropy threshold used before calibration."""
    calibration_tasks: int = 2
    """Number of cold-start tasks used to calibrate N3 entropy threshold."""

    # DBSCAN
    dbscan_eps: float = 0.5
    """DBSCAN epsilon radius in cosine space."""
    dbscan_min_samples: int = 5
    """Minimum points to form a dense DBSCAN region."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NecessityResult:
    """Output of the SOMA-NECESSITY check.

    Contains both boolean signals and continuous scores for the RL state vector.
    """

    necessity: bool
    """True iff N1 AND N2 AND N3 all fire."""

    n1: bool
    """N1 — loss plateau detected."""
    n2: bool
    """N2 — subspace saturation detected."""
    n3: bool
    """N3 — systematic failure detected."""

    # Continuous scores for RL state
    plateau_score: float = 0.0
    """Continuous [0,1]: 0 = training fine, 1 = fully plateaued."""
    residual_fraction: float = 0.0
    """Continuous [0,1]: 0 = in subspace, 1 = fully orthogonal."""
    silhouette: float = 0.0
    """Cluster quality score from DBSCAN."""
    entropy: float = 0.0
    """Cluster entropy — low = systematic, high = random."""
    n_failures: int = 0
    """Number of failure gradients collected."""


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class SomaNecessity:
    """Three-signal necessity detector for SOMA capacity growth.

    Usage::

        nec = SomaNecessity(NecessityConfig())
        nec.reset_for_task()

        for batch in task_data:
            loss = model(batch)
            grad = gradient(loss, adapter_params)
            nec.update_loss(loss)
            nec.add_gradient(grad)
            if model_was_wrong:
                nec.add_failure_gradient(grad)

        result = nec.check()
        if result.necessity:
            # New capacity is genuinely needed
            ...

        nec.task_completed(grads_this_task)
    """

    def __init__(self, cfg: Optional[NecessityConfig] = None) -> None:
        self.cfg = cfg or NecessityConfig()

        # N1 state
        self._loss_history: List[float] = []
        self._best_smoothed: float = float("inf")
        self._stale_steps: int = 0

        # N2 state
        self._gradient_buffer: List[np.ndarray] = []
        self._principal_basis: Optional[np.ndarray] = None  # shape [d, rank]

        # N3 state
        self._failure_gradient_buffer: List[np.ndarray] = []
        self._calibration_entropies: List[float] = []
        self._entropy_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # N1 — Loss Plateau
    # ------------------------------------------------------------------

    def update_loss(self, loss: float) -> bool:
        """Record a training loss value. Returns True if plateau detected.

        Call every training step.
        """
        self._loss_history.append(loss)

        window = self.cfg.plateau_window
        if len(self._loss_history) < window:
            return False

        smoothed = float(np.mean(self._loss_history[-window:]))

        if smoothed < self._best_smoothed - self.cfg.plateau_min_delta:
            self._best_smoothed = smoothed
            self._stale_steps = 0
        else:
            self._stale_steps += 1

        return self._stale_steps >= self.cfg.plateau_patience

    def _check_n1(self) -> tuple[bool, float]:
        """Return (n1_fired, plateau_score)."""
        score = min(1.0, self._stale_steps / self.cfg.plateau_patience)
        fired = self._stale_steps >= self.cfg.plateau_patience
        return fired, score

    # ------------------------------------------------------------------
    # N2 — Subspace Saturation
    # ------------------------------------------------------------------

    def add_gradient(self, g: np.ndarray) -> None:
        """Buffer a gradient vector for N2 subspace analysis.

        Call every training step. ``g`` must be a flat 1-D array of shape ``[d]``.
        """
        self._gradient_buffer.append(g.ravel().astype(np.float32))

    def _check_n2(self) -> tuple[bool, float]:
        """Return (n2_fired, residual_fraction).

        If no principal basis exists yet (first task), returns (False, 0.0).
        """
        if self._principal_basis is None or len(self._gradient_buffer) == 0:
            return False, 0.0

        # Mean gradient for this task
        g_new = np.mean(self._gradient_buffer, axis=0).astype(np.float64)
        g_norm = np.linalg.norm(g_new)
        if g_norm < 1e-12:
            return False, 0.0
        g_new = g_new / g_norm

        P = self._principal_basis.astype(np.float64)  # [d, rank]
        proj = P @ (P.T @ g_new)  # projection onto P
        residual = g_new - proj
        residual_fraction = float(np.dot(residual, residual) / np.dot(g_new, g_new))

        fired = residual_fraction > self.cfg.residual_threshold
        return fired, residual_fraction

    def update_basis(self, grads: List[np.ndarray]) -> None:
        """Update the principal subspace basis after completing a task.

        Uses thin SVD on stacked gradients, then QR-orthogonalises with existing basis.
        """
        if len(grads) == 0:
            return

        G = np.stack([g.ravel().astype(np.float64) for g in grads])
        # Thin SVD
        try:
            _, _, Vt = np.linalg.svd(G, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        rank = min(self.cfg.subspace_rank, Vt.shape[0])
        new_vecs = Vt[:rank].T  # [d, rank]

        if self._principal_basis is None:
            self._principal_basis = new_vecs
        else:
            combined = np.concatenate([self._principal_basis, new_vecs], axis=1)
            Q, _ = np.linalg.qr(combined)
            self._principal_basis = Q[:, :self.cfg.subspace_rank]

    # ------------------------------------------------------------------
    # N3 — Systematic Failure Detection
    # ------------------------------------------------------------------

    def add_failure_gradient(self, g: np.ndarray) -> None:
        """Buffer a failure-case gradient for N3 clustering.

        Call when model prediction is incorrect.
        """
        self._failure_gradient_buffer.append(g.ravel().astype(np.float32))

    def _check_n3(self) -> tuple[bool, float, float, int]:
        """Return (n3_fired, silhouette, entropy, n_failures)."""
        n_fail = len(self._failure_gradient_buffer)
        if n_fail < self.cfg.min_failures:
            return False, 0.0, 0.0, n_fail

        labels, entropy, G_norm = self._run_dbscan(self._failure_gradient_buffer)
        if labels is None or entropy is None or G_norm is None:
            return False, 0.0, 0.0, n_fail

        # Count non-noise clusters
        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return False, 0.0, 0.0, n_fail

        # Silhouette score (only on labelled points)
        mask = labels != -1
        if mask.sum() < 2:
            return False, 0.0, 0.0, n_fail

        try:
            sil = float(silhouette_score(G_norm[mask], labels[mask], metric="cosine"))
        except Exception:
            sil = 0.0

        # Entropy of cluster sizes
        labelled_labels = labels[mask]
        counts = np.bincount(labelled_labels[labelled_labels >= 0])
        p_k = counts / counts.sum()
        p_k = p_k[p_k > 0]
        entropy = float(-np.sum(p_k * np.log(p_k)))

        # Determine threshold
        threshold = self._entropy_threshold
        if threshold is None:
            threshold = self.cfg.entropy_default

        fired = (sil > self.cfg.silhouette_min) and (entropy < threshold)
        return fired, sil, entropy, n_fail

    def _run_dbscan(
        self,
        grads: List[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
        """Run DBSCAN in projected cosine space.

        Returns (labels, entropy, projected_points). Any value may be None on failure.
        """
        if len(grads) < self.cfg.min_failures:
            return None, None, None

        G = np.stack([g.ravel().astype(np.float64) for g in grads])
        G -= G.mean(axis=0)

        try:
            _, _, Vt = np.linalg.svd(G, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, None, None

        proj_dim = min(self.cfg.proj_dim, Vt.shape[0])
        G_proj = G @ Vt[:proj_dim].T
        norms = np.linalg.norm(G_proj, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        G_norm = G_proj / norms

        db = DBSCAN(
            eps=self.cfg.dbscan_eps,
            min_samples=self.cfg.dbscan_min_samples,
            metric="cosine",
        ).fit(G_norm)
        labels = db.labels_

        labelled = labels[labels != -1]
        if len(labelled) == 0:
            return labels, float("inf"), G_norm

        counts = np.bincount(labelled)
        p_k = counts / counts.sum()
        eps = 1e-12
        entropy = float(-np.sum(p_k * np.log(p_k + eps)))

        return labels, entropy, G_norm

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check(self) -> NecessityResult:
        """Run full N1+N2+N3 check.

        Call after sufficient training on the current task.
        Returns :class:`NecessityResult` with all signals and continuous scores.
        """
        n1, plateau_score = self._check_n1()
        n2, residual_fraction = self._check_n2()
        n3, sil, entropy, n_failures = self._check_n3()

        necessity = n1 and n2 and n3

        return NecessityResult(
            necessity=necessity,
            n1=n1,
            n2=n2,
            n3=n3,
            plateau_score=plateau_score,
            residual_fraction=residual_fraction,
            silhouette=sil,
            entropy=entropy,
            n_failures=n_failures,
        )

    def task_completed(self, grads: List[np.ndarray]) -> None:
        """Call after finishing training on a task.

        Updates principal subspace basis and calibrates N3 entropy threshold
        during the cold-start phase (first ``calibration_tasks`` tasks).
        """
        # Update N2 subspace basis
        self.update_basis(grads)

        # Calibrate N3 entropy threshold from cold-start tasks
        if len(self._calibration_entropies) < self.cfg.calibration_tasks:
            _, _, entropy, _ = self._check_n3()
            if entropy > 0:
                self._calibration_entropies.append(entropy)

            if len(self._calibration_entropies) == self.cfg.calibration_tasks:
                mean_entropy = float(np.mean(self._calibration_entropies))
                self._entropy_threshold = self.cfg.entropy_default * mean_entropy

    def reset_for_task(self) -> None:
        """Reset per-task state before starting a new task.

        Clears N1 loss history, N2 gradient buffer, and N3 failure buffer.
        Does NOT clear the principal subspace basis or calibration state.
        """
        self._loss_history.clear()
        self._best_smoothed = float("inf")
        self._stale_steps = 0
        self._gradient_buffer.clear()
        self._failure_gradient_buffer.clear()

    def rl_state_features(self) -> np.ndarray:
        """Return 4 continuous features for the RL state vector.

        Returns:
            np.ndarray of shape [4]:
                [plateau_score, residual_fraction, failure_entropy, failures_norm]
        """
        _, plateau_score = self._check_n1()
        _, residual_fraction = self._check_n2()
        _, _, entropy, n_failures = self._check_n3()
        failures_norm = min(1.0, n_failures / self.cfg.min_failures)
        return np.array(
            [plateau_score, residual_fraction, entropy, failures_norm],
            dtype=np.float32,
        )
