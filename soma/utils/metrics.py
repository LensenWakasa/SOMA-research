"""
SOMA Metrics — BT, FT, per-parameter normalised accuracy.

Precise definitions matching Section 13 of the engineering specification.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Any

import numpy as np


def backward_transfer(
    eval_fn: Callable[[int, Any], float],
    past_task_data: List[Any],
    peak_accuracies: Dict[int, float],
) -> float:
    """Backward Transfer (Definition 2.3).

    BT = (1/(T-1)) * sum_{i=1}^{T-1} [A(M_T, T_i) - A(M_i, T_i)]

    Args:
        eval_fn: eval_fn(adapter_idx=-1, task_data) -> accuracy.
        past_task_data: List of data for all previously completed tasks.
        peak_accuracies: Dict mapping task_idx -> accuracy right after training.

    Returns:
        BT score. 0.0 = perfect retention. Negative = forgetting.
    """
    if len(past_task_data) == 0:
        return 0.0

    diffs = []
    for i, old_data in enumerate(past_task_data):
        current_acc = eval_fn(-1, old_data)
        peak_acc = peak_accuracies.get(i, current_acc)
        diffs.append(current_acc - peak_acc)

    return float(np.mean(diffs))


def forward_transfer(
    eval_fn: Callable[[int, Any], float],
    task_data: Any,
    random_accuracy: float = 0.10,
) -> float:
    """Forward Transfer (Section 13.2).

    FT = A(M_{i-1}, T_i) - A(random, T_i)

    Args:
        eval_fn: eval_fn(adapter_idx=-1, task_data) -> accuracy before training.
        task_data: Data for the task (evaluated BEFORE training on it).
        random_accuracy: Expected random baseline (0.10 for 10-class problems).

    Returns:
        FT score. Positive = beneficial transfer.
    """
    pre_train_acc = eval_fn(-1, task_data)
    return pre_train_acc - random_accuracy


def per_parameter_accuracy(
    accuracy: float,
    n_params: int,
    n_params_baseline: int,
) -> float:
    """Per-Parameter Normalised Accuracy (Section 13.3).

    PPA = accuracy / log(1 + n_params / n_params_baseline)

    Args:
        accuracy: Model accuracy on the evaluation set.
        n_params: Total trainable parameters in SOMA system (K * adapter_size).
        n_params_baseline: Parameter count of the baseline model.

    Returns:
        PPA score. Higher = more parameter-efficient.
    """
    if n_params_baseline <= 0:
        return accuracy
    return accuracy / math.log(1 + n_params / n_params_baseline)


def necessity_precision_recall(
    growth_events: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Necessity Signal Precision/Recall (Section 13.4).

    Precision = TP / (TP + FP) where TP = growth that improved BT
    Recall = TP / (TP + FN) where FN = tasks that needed growth but NECESSITY=False

    Args:
        growth_events: List of dicts with keys:
            - 'necessity': bool (N1 AND N2 AND N3)
            - 'grew': bool (adapter count increased)
            - 'bt_improved': bool (BT improved or stayed within target)

    Returns:
        Dict with 'precision' and 'recall' values.
    """
    tp = fp = fn = 0
    for ev in growth_events:
        if ev.get("grew", False) and ev.get("bt_improved", False):
            tp += 1
        elif ev.get("grew", False) and not ev.get("bt_improved", False):
            fp += 1
        elif not ev.get("grew", False) and not ev.get("bt_improved", False):
            fn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"precision": precision, "recall": recall}


def override_rate(n_forced_merges: int, n_tasks: int) -> float:
    """Override Rate (Section 13.5).

    override_rate = n_forced_merges / n_tasks

    Target: < 0.15. If higher, max_K is too low.
    """
    if n_tasks == 0:
        return 0.0
    return n_forced_merges / n_tasks
