"""
soma/pipeline/loop.py — The Complete SOMA Learning Loop
=========================================================

The full pipeline as described:

  CURIOSITY (always-on)
    → RETRIEVE (always search, even for known things)
    → CLARIFY (ask questions if gaps detected)
    → VERIFY_INTERNAL (what does the model already know?)
    → VERIFY_EXTERNAL (Lean/Z3 for logical consistency)
    → SELF_LEARN (trial and error, minutes not hours)
    → NECESSITY CHECK (N1∧N2∧N3 — only if self-learning insufficient)
    → GROW (spawn adapter) OR ANSWER (if necessity not triggered)

Two learning modes:
  - Skills:    LoRA fine-tuning on specific domain knowledge
  - Reasoning: AlphaProof-style search over solution space

Qwen3.6-27B is the recommended base model:
  - 262K context (Thinking Preservation across loop iterations)
  - Dense 27B, Q4_K_M fits T4 (~16.8GB)
  - Native tool use for retrieval
  - Apache 2.0 license (free)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loop config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopConfig:
    # Base model
    model_name: str = "Qwen/Qwen3.6-27B"
    load_in_4bit: bool = True              # Q4 for T4 VRAM (16.8GB)
    use_thinking_preservation: bool = True # retain reasoning across loop turns

    # Retrieval
    always_search: bool = True             # search even for known things
    max_sources: int = 6

    # Self-learning
    self_learn_steps: int = 100            # quick fine-tune steps (minutes)
    self_learn_lr: float = 2e-4
    self_learn_mode: str = "skills"        # "skills" | "reasoning" | "both"

    # Reasoning search (AlphaProof-style, Phase 2)
    reasoning_depth: int = 8
    reasoning_beam: int = 4

    # Necessity gate
    necessity_enabled: bool = True
    confidence_floor: float = 0.70         # below → self-learn first

    # Adapter
    lora_rank: int = 8                     # overridden by adaptive rank
    max_adapters: int = 20

    # Logging
    log_dir: str = "./soma_logs"


# ─────────────────────────────────────────────────────────────────────────────
# Loop stages enum
# ─────────────────────────────────────────────────────────────────────────────

class LoopStage(str, Enum):
    CURIOSITY        = "curiosity"
    RETRIEVE         = "retrieve"
    CLARIFY          = "clarify"
    VERIFY_INTERNAL  = "verify_internal"
    VERIFY_EXTERNAL  = "verify_external"
    SELF_LEARN       = "self_learn"
    NECESSITY        = "necessity"
    GROW             = "grow"
    ANSWER           = "answer"


# ─────────────────────────────────────────────────────────────────────────────
# Loop result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopResult:
    question: str
    answer: str
    confidence: float
    stages_executed: list[str]
    curiosity_score: float
    did_self_learn: bool
    did_grow: bool
    adapter_spawned: Optional[int]          # adapter index if spawned
    necessity_triggered: bool
    reasoning_mode: str                     # "skills" | "reasoning" | "none"
    sources: list[str]
    discoveries: list[str]
    recommended_rank: Optional[int]         # from adaptive rank
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        return (
            f"Q: {self.question[:80]}\n"
            f"A: {self.answer[:200]}\n"
            f"Confidence: {self.confidence:.0%} | Curiosity: {self.curiosity_score:.3f}\n"
            f"Stages: {' → '.join(self.stages_executed)}\n"
            f"Self-learned: {self.did_self_learn} | Grew: {self.did_grow} | "
            f"Necessity: {self.necessity_triggered}\n"
            f"Rank used: {self.recommended_rank or 'default'}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# The Main Loop
# ─────────────────────────────────────────────────────────────────────────────

class SomaLoop:
    """
    The complete SOMA learning loop.

    Curiosity is always-on. Every question runs through all stages.
    Some stages are fast (curiosity, necessity check = milliseconds).
    Self-learn is the expensive one (~minutes on T4).
    Grow (adapter spawn) is triggered only when self-learn is insufficient.

    Usage:
        loop = SomaLoop(cfg)
        result = await loop.run("What causes the Coriolis effect?")
        print(result.summary())
    """

    def __init__(self, cfg: Optional[LoopConfig] = None):
        self.cfg = cfg or LoopConfig()
        self._init_components()
        self.adapter_pool: list = []
        self.iteration = 0

    def _init_components(self):
        """Lazy-import and initialise all components."""
        from soma.curiosity.engine import CuriosityEngine, CuriosityConfig
        from soma.necessity.engine import SomaNecessity, NecessityConfig
        from soma.retrieval.search import RetrieveAndClarify
        from soma.verification.verifier import Verifier
        from soma.learning.trainer import SelfLearner

        self.curiosity  = CuriosityEngine(CuriosityConfig())
        self.necessity  = SomaNecessity(NecessityConfig())
        self.retriever  = RetrieveAndClarify(always_search=self.cfg.always_search)
        self.verifier   = Verifier()
        self.learner    = SelfLearner(
            steps=self.cfg.self_learn_steps,
            lr=self.cfg.self_learn_lr,
        )

    async def run(self, question: str) -> LoopResult:
        """
        Run the full SOMA loop on a question.
        """
        logger.info(f"[LOOP] Starting: {question[:60]}")
        stages = []
        sources: list[str] = []
        discoveries: list[str] = []
        did_self_learn = False
        did_grow = False
        necessity_triggered = False
        adapter_spawned = None
        reasoning_mode = "none"
        recommended_rank = None

        # ── STAGE 1: CURIOSITY (always-on) ───────────────────────────────
        stages.append(LoopStage.CURIOSITY)
        logger.info("[1/8] CURIOSITY")
        curiosity_signal = await self._run_curiosity(question)
        recommended_rank = curiosity_signal.recommended_rank
        logger.info(f"  C={curiosity_signal.C:.3f} | learnable={curiosity_signal.is_learnable} | threshold={curiosity_signal.adaptive_threshold:.3f}")

        # ── STAGE 2: RETRIEVE (always — even for known things) ────────────
        stages.append(LoopStage.RETRIEVE)
        logger.info("[2/8] RETRIEVE (always-on)")
        # Always search — catches knowledge cutoffs, world-model drift
        retrieval = await self.retriever.run(
            signal=curiosity_signal,
            context=question,
            force=self.cfg.always_search,  # force=True overrides is_learnable gate
        )
        sources.extend(retrieval.get("sources", []))

        # ── STAGE 3: CLARIFY ──────────────────────────────────────────────
        if retrieval.get("clarifying_questions"):
            stages.append(LoopStage.CLARIFY)
            logger.info(f"[3/8] CLARIFY — {len(retrieval['clarifying_questions'])} questions")
            # In production: send clarifying questions, await answers
            # In batch mode: auto-answer from retrieved docs
            # Stub: pass through

        # ── STAGE 4: VERIFY INTERNAL ──────────────────────────────────────
        stages.append(LoopStage.VERIFY_INTERNAL)
        logger.info("[4/8] VERIFY_INTERNAL — checking model knowledge")
        internal_check = await self._verify_internal(
            question=question,
            retrieved_docs=retrieval.get("retrieved_docs", []),
        )
        confidence = internal_check["confidence"]
        logger.info(f"  Internal confidence: {confidence:.0%} | cutoff_detected={internal_check.get('cutoff_detected', False)}")

        # ── STAGE 5: VERIFY EXTERNAL (Lean/Z3) ───────────────────────────
        if internal_check.get("has_logical_claims", False):
            stages.append(LoopStage.VERIFY_EXTERNAL)
            logger.info("[5/8] VERIFY_EXTERNAL — Z3/Lean consistency check")
            external_check = self.verifier.verify(
                statement=internal_check.get("logical_claim", ""),
                retrieved_docs=retrieval.get("retrieved_docs", []),
                use_z3=True,
            )
            # Adjust confidence based on formal verification
            if not external_check["consistent"]:
                confidence *= 0.5
                logger.info("  ⚠ Logical inconsistency detected — confidence halved")

        # ── STAGE 6: SELF-LEARN ───────────────────────────────────────────
        # Triggered when: curiosity is high OR confidence is below floor
        should_self_learn = (
            curiosity_signal.C > self.cfg.confidence_floor or
            confidence < self.cfg.confidence_floor
        )
        pre_curiosity = curiosity_signal

        if should_self_learn:
            stages.append(LoopStage.SELF_LEARN)
            logger.info("[6/8] SELF_LEARN — trial and error fine-tuning")

            # Choose learning mode
            reasoning_mode = self._select_learning_mode(question, curiosity_signal)
            logger.info(f"  Mode: {reasoning_mode}")

            learn_result = await self.learner.learn(
                question=question,
                retrieved_docs=retrieval.get("retrieved_docs", []),
                mode=reasoning_mode,
                lora_rank=recommended_rank or self.cfg.lora_rank,
            )
            did_self_learn = True
            confidence = max(confidence, learn_result.get("confidence_after", confidence))

            # Re-evaluate curiosity after learning
            post_curiosity = await self._run_curiosity(question)
            reduction = self.curiosity.record_outcome(pre_curiosity, post_curiosity)
            logger.info(f"  Epistemic reduction: {reduction:.4f}")

        # ── STAGE 7: NECESSITY CHECK ──────────────────────────────────────
        # Only if confidence still below floor after self-learning
        if self.cfg.necessity_enabled and confidence < self.cfg.confidence_floor:
            stages.append(LoopStage.NECESSITY)
            logger.info("[7/8] NECESSITY CHECK — N1∧N2∧N3")
            nec_result = self.necessity.check()
            necessity_triggered = nec_result.necessity
            logger.info(f"  N1={nec_result.n1} N2={nec_result.n2} N3={nec_result.n3} → NECESSITY={necessity_triggered}")

            if necessity_triggered:
                # ── STAGE 8a: GROW ────────────────────────────────────────
                stages.append(LoopStage.GROW)
                logger.info("[8/8] GROW — spawning new adapter")
                rank = recommended_rank or self.cfg.lora_rank
                adapter_idx = self._spawn_adapter(question, rank)
                adapter_spawned = adapter_idx
                did_grow = True
                logger.info(f"  Adapter spawned: Φ{adapter_idx} (rank={rank})")
        else:
            necessity_triggered = False

        # ── STAGE 8b: ANSWER ──────────────────────────────────────────────
        stages.append(LoopStage.ANSWER)
        answer = await self._generate_answer(
            question=question,
            retrieved_docs=retrieval.get("retrieved_docs", []),
            adapter_idx=adapter_spawned,
        )
        discoveries = self._find_discoveries(answer, retrieval.get("retrieved_docs", []))

        self.iteration += 1

        result = LoopResult(
            question=question,
            answer=answer,
            confidence=confidence,
            stages_executed=[s.value for s in stages],
            curiosity_score=curiosity_signal.C,
            did_self_learn=did_self_learn,
            did_grow=did_grow,
            adapter_spawned=adapter_spawned,
            necessity_triggered=necessity_triggered,
            reasoning_mode=reasoning_mode,
            sources=sources,
            discoveries=discoveries,
            recommended_rank=recommended_rank,
        )

        self._log_result(result)
        return result

    # ── Component implementations ─────────────────────────────────────────

    async def _run_curiosity(self, question: str):
        """
        Evaluate curiosity. In production: run MC dropout on model.
        In stub mode: returns synthetic signal for testing.
        """
        # Stub: synthetic logit samples
        # In production: replace with actual model forward passes with dropout
        rng = np.random.default_rng(hash(question) % (2**32))
        vocab_size = 1000
        logits_samples = [
            rng.normal(0, 1, vocab_size) for _ in range(self.cfg.lora_rank)
        ]
        return self.curiosity.evaluate(logits_samples=logits_samples)

    async def _verify_internal(self, question: str, retrieved_docs: list) -> dict:
        """Check internal model knowledge vs retrieved world knowledge."""
        # In production: compare model's prior answer with retrieved docs
        # Detect knowledge cutoffs by checking if model's answer contradicts docs
        confidence = 0.70 if retrieved_docs else 0.40
        has_logical = any(kw in question.lower() for kw in
                          ["prove", "therefore", "must", "if", "then"])
        return {
            "confidence": confidence,
            "cutoff_detected": False,
            "has_logical_claims": has_logical,
            "logical_claim": question if has_logical else "",
        }

    def _select_learning_mode(self, question: str, signal) -> str:
        """
        Choose between Skills (LoRA) and Reasoning (AlphaProof).

        Skills mode: domain knowledge, factual gaps, language tasks
        Reasoning mode: mathematical proofs, multi-step logic, code

        Heuristic: if curiosity gap is in the logit direction (model was
        confident but wrong) → reasoning gap → AlphaProof mode.
        If gap is entropy-based (model didn't know) → knowledge gap → LoRA.
        """
        if self.cfg.self_learn_mode == "skills":
            return "skills"
        if self.cfg.self_learn_mode == "reasoning":
            return "reasoning"
        # Auto-select
        q = question.lower()
        reasoning_keywords = ["prove", "derive", "calculate", "solve", "code",
                               "algorithm", "formula", "theorem", "step by step"]
        if any(kw in q for kw in reasoning_keywords):
            return "reasoning"
        return "skills"

    def _spawn_adapter(self, question: str, rank: int) -> int:
        """Spawn a new LoRA adapter and add it to the pool."""
        adapter_idx = len(self.adapter_pool)
        # Stub: zero-initialised B, random A (real: train on self-learn data)
        B = np.zeros((128, rank))
        A = np.random.normal(0, 0.01, (rank, 128))
        self.adapter_pool.append({"B": B, "A": A, "rank": rank,
                                   "domain": question[:50], "frozen": False})
        return adapter_idx

    def _freeze_adapter(self, idx: int):
        """Freeze adapter after training — forgetting now impossible."""
        if idx < len(self.adapter_pool):
            self.adapter_pool[idx]["frozen"] = True

    async def _generate_answer(
        self, question: str, retrieved_docs: list, adapter_idx: Optional[int]
    ) -> str:
        """Generate final answer using base model + adapter if available."""
        # Stub: in production, run model with retrieved context + adapter
        context = "\n".join(retrieved_docs[:3]) if retrieved_docs else ""
        adapter_note = f" [Adapter Φ{adapter_idx}]" if adapter_idx is not None else ""
        return (
            f"[SOMA Answer{adapter_note}] Based on {len(retrieved_docs)} retrieved "
            f"sources: {question[:60]}... "
            f"{'Context: ' + context[:200] if context else 'No external sources retrieved.'}"
        )

    def _find_discoveries(self, answer: str, docs: list) -> list[str]:
        """Identify claims in answer not present in retrieved docs."""
        # Stub: full implementation uses semantic similarity check
        return []

    def _log_result(self, result: LoopResult):
        """Save result to log file."""
        import os
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        path = f"{self.cfg.log_dir}/loop_{self.iteration:05d}.json"
        with open(path, "w") as f:
            d = {k: v for k, v in result.__dict__.items()
                 if not isinstance(v, np.ndarray)}
            json.dump(d, f, indent=2)
