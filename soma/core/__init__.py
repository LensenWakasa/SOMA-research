"""
SOMA Core — Algorithm implementations.

Exports:
    SomaNecessity   — N1+N2+N3 necessity detector
    SomaGrow        — RL policy with 4 growth actions
    SomaRouter      — Prototype-based adapter router
    SomaLearn       — Outer training loop
"""

from soma.core.necessity import SomaNecessity, NecessityConfig, NecessityResult
from soma.core.grow import SomaGrow, GrowConfig, GrowResult, GrowthPolicy
from soma.core.router import SomaRouter
from soma.core.learn import SomaLearn, LearnConfig, TaskLog

__all__ = [
    "SomaNecessity", "NecessityConfig", "NecessityResult",
    "SomaGrow", "GrowConfig", "GrowResult", "GrowthPolicy",
    "SomaRouter",
    "SomaLearn", "LearnConfig", "TaskLog",
]
