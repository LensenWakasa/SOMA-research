"""
SOMA-GROW — RL-guided capacity growth controller.

Implements:
    - GrowthPolicy: Linear REINFORCE policy (W: [7,4])
    - SomaGrow: Action selection + execution + reward computation

Four actions:
    0 = UPDATE_EXISTING — retrain existing adapter with KL gating
    1 = SPAWN_NEW — create fresh adapter
    2 = MERGE — combine two most-similar adapters
    3 = SKIP — do nothing

The policy learns from reward signals over time, using REINFORCE
with normalised returns updated every 5 tasks.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class GrowAction(enum.IntEnum):
    UPDATE_EXISTING = 0
    SPAWN_NEW = 1
    MERGE = 2
    SKIP = 3


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GrowConfig:
    """Configuration for the SOMA-GROW controller.

    All defaults match Section 6.3 of the engineering specification.
    """

    alpha: float = 1.0
    """Reward weight for new-task accuracy gain."""
    beta: float = 2.0
    """Penalty weight for backward transfer (forgetting). beta > alpha = conservative."""
    gamma: float = 0.5
    """Penalty weight for adapter count increase."""
    kl_gate: float = 0.10
    """Maximum KL proxy for UPDATE gating."""
    max_k: int = 20
    """Growth ceiling. Forces MERGE when K reaches max_k."""
    policy_lr: float = 0.01
    """Learning rate for REINFORCE policy updates."""
    discount: float = 0.99
    """Discount factor for return computation."""
    update_every: int = 5
    """Update policy every N tasks."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GrowResult:
    """Output of a SOMA-GROW step."""

    action: GrowAction
    """The action taken."""
    reward: float
    """Computed reward for this step."""
    k_before: int
    """Adapter count before action."""
    k_after: int
    """Adapter count after action."""
    action_name: str = ""
    """Human-readable action name."""


# ---------------------------------------------------------------------------
# GrowthPolicy — Linear REINFORCE
# ---------------------------------------------------------------------------


class GrowthPolicy:
    """Linear policy trained by REINFORCE.

    State vector (7 features) -> action probabilities (4 actions).
    W shape: [7, 4], b shape: [4].

    Initialisation: W ~ Normal(0, 0.01), b = zeros.
    This near-zero init means all actions start equally likely.
    """

    N_FEATURES = 7
    N_ACTIONS = 4

    def __init__(self, lr: float = 0.01, discount: float = 0.99) -> None:
        self.lr = lr
        self.discount = discount

        # Near-zero init — all actions start equally likely
        self.W = np.random.randn(self.N_FEATURES, self.N_ACTIONS).astype(np.float64) * 0.01
        self.b = np.zeros(self.N_ACTIONS, dtype=np.float64)

        # Trajectory buffer
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = logits - logits.max()
        e = np.exp(x)
        return e / e.sum()

    def select(self, state: np.ndarray) -> GrowAction:
        """Select an action given the state vector.

        Args:
            state: shape [7] normalised state features.

        Returns:
            Selected :class:`GrowAction`.
        """
        state = np.asarray(state, dtype=np.float64)
        logits = state @ self.W + self.b
        probs = self._softmax(logits)
        action_idx = int(np.random.choice(self.N_ACTIONS, p=probs))
        return GrowAction(action_idx)

    def action_probs(self, state: np.ndarray) -> np.ndarray:
        """Return action probability distribution for a state."""
        state = np.asarray(state, dtype=np.float64)
        logits = state @ self.W + self.b
        return self._softmax(logits)

    def record(self, state: np.ndarray, action: int, reward: float) -> None:
        """Buffer a (state, action, reward) transition for later REINFORCE update."""
        self._states.append(np.asarray(state, dtype=np.float64))
        self._actions.append(int(action))
        self._rewards.append(float(reward))

    def update(self) -> float:
        """Run REINFORCE gradient update on the buffered trajectory.

        Returns:
            Mean reward of the trajectory.
        """
        if len(self._rewards) == 0:
            return 0.0

        # Compute discounted returns
        T = len(self._rewards)
        returns = np.zeros(T, dtype=np.float64)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.discount * G
            returns[t] = G

        # Normalise returns
        if T > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE policy gradient
        for t in range(T):
            s_t = self._states[t]
            a_t = self._actions[t]
            G_t = returns[t]

            logits = s_t @ self.W + self.b
            probs = self._softmax(logits)

            # Policy gradient
            grad_logits = probs.copy()
            grad_logits[a_t] -= 1.0
            grad_logits *= G_t * self.lr

            self.W -= np.outer(s_t, grad_logits)
            self.b -= grad_logits

        mean_reward = float(np.mean(self._rewards))

        # Clear buffer
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()

        return mean_reward

    def get_weights(self) -> dict:
        """Return policy weights for checkpointing."""
        return {"W": self.W.copy(), "b": self.b.copy()}

    def load_weights(self, weights: dict) -> None:
        """Load policy weights from checkpoint."""
        self.W = weights["W"].copy()
        self.b = weights["b"].copy()


# ---------------------------------------------------------------------------
# SomaGrow — Growth Controller
# ---------------------------------------------------------------------------


class SomaGrow:
    """RL-guided capacity growth controller.

    Receives the necessity signal and current system state, then selects
    and executes one of four actions: UPDATE_EXISTING, SPAWN_NEW, MERGE, or SKIP.

    Usage::

        grow = SomaGrow(GrowConfig())
        result = grow.step(
            state=state_vector,
            nec_result=nec_result,
            pool=adapter_pool,
            task_data=task_data,
            past_task_data=past_task_data,
            train_fn=train_fn,
            eval_fn=eval_fn,
        )
    """

    def __init__(self, cfg: Optional[GrowConfig] = None) -> None:
        self.cfg = cfg or GrowConfig()
        self.policy = GrowthPolicy(lr=self.cfg.policy_lr, discount=self.cfg.discount)
        self._task_count: int = 0

    def build_state_vector(
        self,
        nec_features: np.ndarray,
        k: int,
        max_k: int,
        router_confidence: float,
        steps_since_spawn: int,
    ) -> np.ndarray:
        """Build the 7-dimensional state vector for the policy.

        Args:
            nec_features: [4] from SomaNecessity.rl_state_features()
                [plateau_score, residual_fraction, failure_entropy, failures_norm]
            k: Current adapter count.
            max_k: Maximum adapter count.
            router_confidence: Highest cosine similarity from the router.
            steps_since_spawn: Tasks since last spawn event.

        Returns:
            np.ndarray of shape [7] with all features normalised to [0, 1].
        """
        return np.array([
            nec_features[0],                            # s[0] plateau_score
            nec_features[1],                            # s[1] residual_fraction
            nec_features[2],                            # s[2] failure_entropy
            nec_features[3],                            # s[3] n_failures / min_failures
            min(1.0, k / max(1, max_k)),                # s[4] pool fullness
            min(1.0, router_confidence),                # s[5] router_max_confidence
            min(1.0, steps_since_spawn / 20.0),         # s[6] recency of last spawn
        ], dtype=np.float64)

    def step(
        self,
        state: np.ndarray,
        nec_result,
        pool: list,
        task_data,
        past_task_data: list,
        train_fn: Callable,
        eval_fn: Callable,
        force_action: Optional[GrowAction] = None,
    ) -> GrowResult:
        """Execute one SOMA-GROW step.

        Args:
            state: 7-d state vector from :meth:`build_state_vector`.
            nec_result: :class:`NecessityResult` from SOMA-NECESSITY.
            pool: List of (B, A) adapter tuples.
            task_data: Data for the current task.
            past_task_data: List of data for past tasks. 
            train_fn: ``train_fn(adapter_idx_or_None, task_data, lr_scale) -> (B, A)``
            eval_fn: ``eval_fn(adapter_idx, task_data) -> accuracy``
            force_action: If set, override policy selection (e.g., forced MERGE at ceiling).

        Returns:
            :class:`GrowResult` with action, reward, and adapter count changes.
        """
        k_before = len(pool)

        # Action selection
        if force_action is not None:
            action = force_action
        elif k_before >= self.cfg.max_k:
            action = GrowAction.MERGE
        else:
            action = self.policy.select(state)

        # Evaluate accuracy before action (current task + past tasks)
        acc_before_new = eval_fn(-1, task_data) if k_before > 0 else 0.0
        acc_before_old = []
        for old_data in past_task_data:
            acc_before_old.append(eval_fn(-1, old_data))

        # Execute action
        if action == GrowAction.UPDATE_EXISTING:
            self._execute_update(pool, task_data, train_fn)
        elif action == GrowAction.SPAWN_NEW:
            self._execute_spawn(pool, task_data, train_fn)
        elif action == GrowAction.MERGE:
            self._execute_merge(pool)
        else:  # SKIP
            pass

        k_after = len(pool)

        # Evaluate accuracy after action
        acc_after_new = eval_fn(-1, task_data) if k_after > 0 else 0.0
        acc_after_old = []
        for old_data in past_task_data:
            acc_after_old.append(eval_fn(-1, old_data))

        # Compute reward
        delta_acc_new = acc_after_new - acc_before_new
        if len(acc_before_old) > 0:
            delta_BT = float(np.mean([
                acc_after_old[i] - acc_before_old[i]
                for i in range(len(acc_before_old))
            ]))
        else:
            delta_BT = 0.0
        delta_K = k_after - k_before

        reward = (
            self.cfg.alpha * delta_acc_new
            - self.cfg.beta * abs(delta_BT)
            - self.cfg.gamma * max(0, delta_K)
        )

        # Record for REINFORCE
        self.policy.record(state, int(action), reward)

        self._task_count += 1

        return GrowResult(
            action=action,
            reward=reward,
            k_before=k_before,
            k_after=k_after,
            action_name=action.name,
        )

    def update_policy(self) -> float:
        """Run REINFORCE update on buffered trajectory. Returns mean reward."""
        return self.policy.update()

    def should_update_policy(self) -> bool:
        """Whether enough tasks have passed for a policy update."""
        return self._task_count > 0 and self._task_count % self.cfg.update_every == 0

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _execute_update(self, pool: list, task_data, train_fn: Callable) -> None:
        """UPDATE_EXISTING: retrain best-matching adapter with KL gating."""
        if len(pool) == 0:
            return

        best_idx = self._best_matching_adapter(pool)
        B_old, A_old = pool[best_idx]
        old_norm = np.linalg.norm(B_old @ A_old)

        # First attempt at full learning rate
        B_new, A_new = train_fn(best_idx, task_data, 1.0)
        kl = np.linalg.norm(B_new @ A_new - B_old @ A_old) / max(old_norm, 1e-12)

        if kl < self.cfg.kl_gate:
            pool[best_idx] = (B_new, A_new)
            return

        # Second attempt at half learning rate
        B_new2, A_new2 = train_fn(best_idx, task_data, 0.5)
        kl2 = np.linalg.norm(B_new2 @ A_new2 - B_old @ A_old) / max(old_norm, 1e-12)

        if kl2 < self.cfg.kl_gate:
            pool[best_idx] = (B_new2, A_new2)
        # else: reject entirely — adapter unchanged

    def _execute_spawn(self, pool: list, task_data, train_fn: Callable) -> None:
        """SPAWN_NEW: create a fresh adapter."""
        B_new, A_new = train_fn(None, task_data, 1.0)
        pool.append((B_new, A_new))

    def _execute_merge(self, pool: list) -> None:
        """MERGE: combine two most-similar adapters via averaging.

        Note: Average merge is a placeholder (Problem 2 — partial solution).
        Fisher-weighted or TIES merging would be better.
        """
        if len(pool) < 2:
            return

        # Find most-similar pair by cosine similarity of B@A
        best_sim = -1.0
        best_i, best_j = 0, 1
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                W_i = pool[i][0] @ pool[i][1]
                W_j = pool[j][0] @ pool[j][1]
                sim = float(np.dot(W_i.ravel(), W_j.ravel()) / (
                    np.linalg.norm(W_i) * np.linalg.norm(W_j) + 1e-12
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_i, best_j = i, j

        # Average merge
        B_merged = (pool[best_i][0] + pool[best_j][0]) / 2.0
        A_merged = (pool[best_i][1] + pool[best_j][1]) / 2.0
        pool[best_i] = (B_merged, A_merged)
        pool.pop(best_j)

    def _best_matching_adapter(self, pool: list) -> int:
        """Find the adapter index with minimum weight norm (simple heuristic).

        In practice, the router should identify the best candidate.
        """
        if len(pool) == 1:
            return 0
        norms = [np.linalg.norm(B @ A) for B, A in pool]
        return int(np.argmin(norms))
