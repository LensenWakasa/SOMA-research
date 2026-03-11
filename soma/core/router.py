"""
SOMA Router — Prototype-based nearest-neighbour adapter routing.

Stores ``n_prototypes`` embedding vectors per adapter from that adapter's
training task. Routes new inputs by cosine similarity to stored prototypes.
No gradient updates means no forgetting.

Design rationale:
    A trained classifier would suffer catastrophic forgetting when new adapters
    are added (its output head would need to expand). Prototypes don't. Adding
    a new adapter just adds a new entry to the prototype dictionary — existing
    entries are untouched.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class SomaRouter:
    """Prototype-based adapter router.

    Usage::

        router = SomaRouter(n_prototypes=10)
        router.register(adapter_idx=0, embeddings=task0_embeddings)
        router.register(adapter_idx=1, embeddings=task1_embeddings)

        idx, confidence = router.route(query_embedding)
    """

    def __init__(self, n_prototypes: int = 10) -> None:
        self.n_prototypes = n_prototypes
        # adapter_idx -> np.ndarray of shape [n_prototypes, embed_dim]
        self._prototypes: Dict[int, np.ndarray] = {}

    @property
    def n_adapters(self) -> int:
        """Number of registered adapters."""
        return len(self._prototypes)

    def register(self, adapter_idx: int, embeddings: np.ndarray) -> None:
        """Register prototypes for an adapter.

        Args:
            adapter_idx: Integer index of the adapter.
            embeddings: shape ``[n, embed_dim]``. Up to ``n_prototypes``
                are randomly sampled and stored.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        n = embeddings.shape[0]
        if n > self.n_prototypes:
            indices = np.random.choice(n, self.n_prototypes, replace=False)
            embeddings = embeddings[indices]

        # L2-normalise for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms

        self._prototypes[adapter_idx] = embeddings

    def route(self, query: np.ndarray) -> Tuple[int, float]:
        """Route a single input to the best adapter.

        Args:
            query: shape ``[embed_dim]`` embedding of the input.

        Returns:
            Tuple of (adapter_idx, confidence) where confidence is the
            mean cosine similarity to the best adapter's prototypes.

        Raises:
            RuntimeError: If no adapters are registered.
        """
        if len(self._prototypes) == 0:
            raise RuntimeError("No adapters registered in router.")

        query = np.asarray(query, dtype=np.float32).ravel()
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-12:
            # Zero vector — route to first adapter with low confidence
            return min(self._prototypes.keys()), 0.0
        query = query / q_norm

        best_idx = -1
        best_sim = -float("inf")

        for idx, protos in self._prototypes.items():
            # Cosine similarity: protos are already normalised, query is normalised
            sims = protos @ query  # [n_prototypes]
            mean_sim = float(sims.mean())
            if mean_sim > best_sim:
                best_sim = mean_sim
                best_idx = idx

        return best_idx, max(0.0, best_sim)

    def route_batch(self, queries: np.ndarray) -> List[Tuple[int, float]]:
        """Route a batch of inputs.

        Args:
            queries: shape ``[batch, embed_dim]``.

        Returns:
            List of (adapter_idx, confidence) tuples.
        """
        return [self.route(q) for q in queries]

    def max_confidence(self, query: np.ndarray) -> float:
        """Return the highest cosine similarity for routing a query.

        Used as feature s[5] in the RL state vector.
        """
        if len(self._prototypes) == 0:
            return 0.0
        _, conf = self.route(query)
        return conf

    def remove(self, adapter_idx: int) -> None:
        """Remove an adapter's prototypes (e.g., after a merge).

        Also re-indexes remaining adapters to maintain contiguity.
        """
        if adapter_idx in self._prototypes:
            del self._prototypes[adapter_idx]

        # Re-index to keep adapter indices contiguous [0, 1, 2, ...]
        old_entries = sorted(self._prototypes.items())
        self._prototypes.clear()
        for new_idx, (_, protos) in enumerate(old_entries):
            self._prototypes[new_idx] = protos

    def get_all_prototypes(self) -> Dict[int, np.ndarray]:
        """Return all stored prototypes (for checkpointing)."""
        return {k: v.copy() for k, v in self._prototypes.items()}

    def load_prototypes(self, prototypes: Dict[int, np.ndarray]) -> None:
        """Load prototypes from a checkpoint."""
        self._prototypes = {k: v.copy() for k, v in prototypes.items()}
