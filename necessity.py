"""
SOMA-NECESSITY
==============
Three-signal conjunction gate that decides whether new LoRA capacity
is genuinely needed.

    NECESSITY = N1 AND N2 AND N3

Each signal rules out a distinct false-positive class:

    N1 — Loss Plateau        : rules out noisy loss spikes
    N2 — Subspace Saturation : rules out tasks already covered by existing adapters
    N3 — Systematic Failure  : rules out stochastic noise that will resolve with training

All three must be TRUE simultaneously before any growth occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NecessityConfig:
    # N1 — Loss Plateau
    plateau_patience: int = 300
    """Steps without improvement before N1=True."""
    plateau_min_delta: float = 1e-3
    """Minimum meaningful loss improvement."""
    plateau_window: int = 30
    """Rolling window for loss smoothing."""

    # N2 — Subspace Saturation
    subspace_rank: int = 16
    """Number of principal SVD directions to keep."""
    residual_threshold: float = 0.80
    """Fraction of gradient energy outside existing subspace to declare saturation."""

    # N3 — Systematic Failure
    min_failures: int = 25
    """Minimum failure cases before clustering. Below this, N3=False."""
    proj_dim: int = 32
    """Gradient projection dimension for DBSCAN. 32 is fast and sufficient."""
    silhouette_min: float = 0.30
    """Minimum silhouette score for clusters to be real."""
    entropy_default: float = 0.70
    """Fallback entropy threshold before calibration."""
    calibration_tasks: int = 2
    """Cold-start tasks used to calibrate N3 entropy threshold."""

    # DBSCAN
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NecessityResult:
    necessity: bool
    n1: bool
    n2: bool
    n3: bool

    # Continuous scores (for RL state vector)
    plateau_score: float = 0.0
    residual_fraction: float = 0.0
    silhouette: float = 0.0
    entropy: float = 0.0

    # Diagnostics
    n_failure_grads: int = 0
    n_clusters: int = 0
    stale_steps: int = 0

    def rl_state_features(self) -> np.ndarray:
        """4-feature vector for SOMA-GROW state input."""
        return np.array([
            self.plateau_score,
            self.residual_fraction,
            self.entropy,
            min(self.n_failure_grads / 25.0, 1.0),
        ], dtype=np.float32)

    def __repr__(self) -> str:
        flag = lambda b: "✓" if b else "✗"
        return (
            f"NecessityResult("
            f"NECESSITY={flag(self.necessity)} | "
            f"N1={flag(self.n1)} plateau={self.plateau_score:.2f} | "
            f"N2={flag(self.n2)} residual={self.residual_fraction:.2f} | "
            f"N3={flag(self.n3)} sil={self.silhouette:.2f} ent={self.entropy:.2f}"
            f")"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SomaNecessity:
    """
    Runs continuously during training on a new task.

    Monitors:
      N1 — loss trajectory (has training genuinely stalled?)
      N2 — gradient subspace (is the new task outside existing capacity?)
      N3 — failure clustering (are failures systematic or random noise?)

    All three must be TRUE simultaneously (conjunction gate) before
    growth is authorised.
    """

    def __init__(self, cfg: Optional[NecessityConfig] = None) -> None:
        self.cfg = cfg or NecessityConfig()

        # N1 state
        self._loss_history: List[float] = []
        self._best_smoothed: float = float("inf")
        self._stale_steps: int = 0

        # N2 state — principal subspace basis, shape [d, rank]
        self._subspace_basis: Optional[np.ndarray] = None
        self._gradient_buffer: List[np.ndarray] = []

        # N3 state
        self._failure_grads: List[np.ndarray] = []
        self._calibrated_entropy_threshold: Optional[float] = None
        self._calibration_entropies: List[float] = []

    # ------------------------------------------------------------------
    # Per-step updates (call every training step)
    # ------------------------------------------------------------------

    def update_loss(self, loss: float) -> bool:
        """
        Update N1 plateau detector.

        Returns True if plateau is detected.
        """
        self._loss_history.append(loss)

        window = self.cfg.plateau_window
        if len(self._loss_history) < window:
            self._stale_steps = 0
            return False

        smoothed = float(np.mean(self._loss_history[-window:]))

        if self._best_smoothed - smoothed < self.cfg.plateau_min_delta:
            self._stale_steps += 1
        else:
            self._best_smoothed = smoothed
            self._stale_steps = 0

        return self._stale_steps >= self.cfg.plateau_patience

    def add_gradient(self, g: np.ndarray) -> None:
        """Buffer gradient for N2 subspace check. Shape must be flat [d]."""
        self._gradient_buffer.append(g.ravel().astype(np.float32))

    def add_failure_gradient(self, g: np.ndarray) -> None:
        """Buffer gradient for N3 clustering. Call when prediction is wrong."""
        self._failure_grads.append(g.ravel().astype(np.float32))

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def reset_for_task(self) -> None:
        """
        Call before starting a new task.
        Clears N1 history, N2 gradient buffer, N3 failure buffer.
        Does NOT clear the subspace basis (accumulated across tasks).
        """
        self._loss_history = []
        self._best_smoothed = float("inf")
        self._stale_steps = 0
        self._gradient_buffer = []
        self._failure_grads = []

    def task_completed(self, task_grads: List[np.ndarray]) -> None:
        """
        Call after finishing training on a task.

        (a) Updates principal subspace basis with this task's gradients.
        (b) Handles N3 entropy calibration for first N cold-start tasks.
        """
        self._update_subspace(task_grads)

        # N3 calibration: collect entropy from cold-start tasks
        if len(self._calibration_entropies) < self.cfg.calibration_tasks:
            _, ent = self._run_dbscan(task_grads)
            if ent is not None:
                self._calibration_entropies.append(ent)

            if len(self._calibration_entropies) == self.cfg.calibration_tasks:
                self._calibrated_entropy_threshold = (
                    self.cfg.entropy_default
                    * float(np.mean(self._calibration_entropies))
                )

    # ------------------------------------------------------------------
    # Full conjunction check
    # ------------------------------------------------------------------

    def check(self) -> NecessityResult:
        """
        Run full N1+N2+N3 check. Call after sufficient training steps.
        Returns NecessityResult with all signals and continuous scores.
        """
        n1, plateau_score, stale_steps = self._check_n1()
        n2, residual_fraction = self._check_n2()
        n3, sil, entropy, n_clusters = self._check_n3()

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
            n_failure_grads=len(self._failure_grads),
            n_clusters=n_clusters,
            stale_steps=stale_steps,
        )

    # ------------------------------------------------------------------
    # N1 — Loss Plateau
    # ------------------------------------------------------------------

    def _check_n1(self) -> Tuple[bool, float, int]:
        """
        N1=True iff training has genuinely stalled for plateau_patience steps.

        Returns (n1, plateau_score, stale_steps).
        plateau_score = min(1.0, stale_steps / plateau_patience)
        """
        plateau_score = min(1.0, self._stale_steps / self.cfg.plateau_patience)
        n1 = self._stale_steps >= self.cfg.plateau_patience
        return n1, plateau_score, self._stale_steps

    # ------------------------------------------------------------------
    # N2 — Subspace Saturation
    # ------------------------------------------------------------------

    def _check_n2(self) -> Tuple[bool, float]:
        """
        N2=True iff the new task's gradient is largely orthogonal to the
        existing principal subspace — meaning existing adapters can't help.

        Returns (n2, residual_fraction).
        residual_fraction = ||g - P(P^T g)||^2 / ||g||^2
        """
        if self._subspace_basis is None or len(self._gradient_buffer) == 0:
            # No prior task completed → no basis → cannot saturate
            return False, 0.0

        g_new = np.mean(self._gradient_buffer, axis=0).astype(np.float32)
        norm = np.linalg.norm(g_new)
        if norm < 1e-10:
            return False, 0.0

        g_new = g_new / norm
        P = self._subspace_basis  # [d, rank]

        proj = P @ (P.T @ g_new)   # projection onto subspace
        residual = g_new - proj
        residual_fraction = float(np.dot(residual, residual))

        n2 = residual_fraction > self.cfg.residual_threshold
        return n2, residual_fraction

    def _update_subspace(self, task_grads: List[np.ndarray]) -> None:
        """
        Update principal subspace basis after completing a task.
        Uses thin SVD and QR orthogonalisation.
        """
        if len(task_grads) == 0:
            return

        G = np.stack([g.ravel().astype(np.float32) for g in task_grads])  # [n, d]
        try:
            _, _, Vt = np.linalg.svd(G, full_matrices=False)
        except np.linalg.LinAlgError:
            return

        rank = self.cfg.subspace_rank
        new_vecs = Vt[:rank].T  # [d, rank]

        if self._subspace_basis is None:
            combined = new_vecs
        else:
            combined = np.concatenate([self._subspace_basis, new_vecs], axis=1)

        # QR orthogonalise and keep top `rank` directions
        Q, _ = np.linalg.qr(combined)
        self._subspace_basis = Q[:, :rank]

    # ------------------------------------------------------------------
    # N3 — Systematic Failure Detection
    # ------------------------------------------------------------------

    def _check_n3(self) -> Tuple[bool, float, float, int]:
        """
        N3=True iff failure-case gradients cluster consistently.
        Clustering = failures share a single structural cause (learnable gap).
        No clustering = random noise (don't grow, give it more training).

        Returns (n3, silhouette, entropy, n_clusters).
        """
        if len(self._failure_grads) < self.cfg.min_failures:
            return False, 0.0, float("inf"), 0

        labels, entropy = self._run_dbscan(self._failure_grads)
        if labels is None or entropy is None:
            return False, 0.0, float("inf"), 0

        # Count non-noise clusters
        unique = set(labels) - {-1}
        n_clusters = len(unique)
        if n_clusters < 2:
            return False, 0.0, entropy, 0

        # Silhouette score (cluster tightness)
        mask = labels != -1
        if mask.sum() < 2:
            return False, 0.0, entropy, n_clusters

        G_fail = self._project_failure_grads()
        try:
            sil = float(silhouette_score(G_fail[mask], labels[mask], metric="cosine"))
        except Exception:
            sil = 0.0

        # Entropy threshold: use calibrated if available
        threshold = (
            self._calibrated_entropy_threshold
            if self._calibrated_entropy_threshold is not None
            else self.cfg.entropy_default
        )

        n3 = (sil > self.cfg.silhouette_min) and (entropy < threshold)
        return n3, sil, entropy, n_clusters

    def _project_failure_grads(self) -> np.ndarray:
        """
        Project failure gradients into low-dimensional cosine space for DBSCAN.
        Uses thin SVD projection to proj_dim dimensions.
        """
        G = np.stack(self._failure_grads)           # [n_fail, d]
        G -= np.mean(G, axis=0)                     # centre

        try:
            _, _, Vt = np.linalg.svd(G, full_matrices=False)
        except np.linalg.LinAlgError:
            return G

        dim = min(self.cfg.proj_dim, Vt.shape[0])
        G_proj = G @ Vt[:dim].T                     # [n_fail, proj_dim]

        norms = np.linalg.norm(G_proj, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return G_proj / norms                       # cosine-normalised

    def _run_dbscan(
        self, grads: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Run DBSCAN on gradient list.
        Returns (labels, entropy). entropy=None if clustering fails.
        """
        if len(grads) < self.cfg.min_failures:
            return None, None

        G = np.stack([g.ravel().astype(np.float32) for g in grads])
        G -= np.mean(G, axis=0)

        try:
            _, _, Vt = np.linalg.svd(G, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, None

        dim = min(self.cfg.proj_dim, Vt.shape[0])
        G_proj = G @ Vt[:dim].T
        norms = np.linalg.norm(G_proj, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        G_norm = G_proj / norms

        db = DBSCAN(
            eps=self.cfg.dbscan_eps,
            min_samples=self.cfg.dbscan_min_samples,
            metric="cosine",
        ).fit(G_norm)
        labels = db.labels_

        # Compute cluster entropy (concentration of labels)
        labelled = labels[labels != -1]
        if len(labelled) == 0:
            return labels, float("inf")

        counts = np.bincount(labelled)
        p_k = counts / counts.sum()
        eps = 1e-12
        entropy = float(-np.sum(p_k * np.log(p_k + eps)))

        return labels, entropy
