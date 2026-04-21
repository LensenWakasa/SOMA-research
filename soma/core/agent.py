"""
acsis/core/agent.py
====================
The Acsis Intelligence Agent — main orchestration loop.

One task: understand the universe and help solve human problems.

Loop:
  RETRIEVE → VERIFY → REASON → EXPERIMENT → DISCOVER → GROW → REPEAT
"""
from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pathlib import Path

from acsis.core.config import AcsisConfig
from acsis.tools.research import ResearchTool
from acsis.tools.executor import CodeExecutor
from acsis.memory.knowledge_graph import KnowledgeGraph
from acsis.memory.vector_store import VectorStore
from acsis.core.reasoner import Reasoner
from acsis.interface.notifier import Notifier

logger = logging.getLogger(__name__)


@dataclass
class ThinkResult:
    """Output of one full Acsis reasoning cycle."""
    question: str
    answer: str
    confidence: float          # [0, 1] — how sure Acsis is
    sources: list[str]
    reasoning_chain: list[str] # step-by-step
    discoveries: list[str]     # genuinely new conclusions
    experiments_run: int
    knowledge_added: int       # new facts stored
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "sources": self.sources,
            "reasoning_chain": self.reasoning_chain,
            "discoveries": self.discoveries,
            "experiments_run": self.experiments_run,
            "knowledge_added": self.knowledge_added,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        lines = [
            f"ACSIS RESULT — {self.timestamp}",
            f"{'='*60}",
            f"Question: {self.question}",
            f"",
            f"Answer (confidence: {self.confidence:.0%}):",
            f"  {self.answer}",
            f"",
        ]
        if self.reasoning_chain:
            lines += ["Reasoning:", *[f"  {i+1}. {s}" for i,s in enumerate(self.reasoning_chain)], ""]
        if self.discoveries:
            lines += ["Discoveries:", *[f"  ★ {d}" for d in self.discoveries], ""]
        lines += [
            f"Sources: {len(self.sources)} | Experiments: {self.experiments_run} | Knowledge added: {self.knowledge_added}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class AcsisAgent:
    """
    The Acsis Intelligence Agent.

    Usage:
        agent = AcsisAgent()
        result = await agent.think("What causes malaria and how can we eliminate it?")
        print(result.summary())
    """

    def __init__(self, cfg: Optional[AcsisConfig] = None):
        self.cfg = cfg or AcsisConfig()
        self.research  = ResearchTool(cfg=self.cfg)
        self.executor  = CodeExecutor(cfg=self.cfg)
        self.memory    = VectorStore(cfg=self.cfg)
        self.graph     = KnowledgeGraph(cfg=self.cfg)
        self.reasoner  = Reasoner(cfg=self.cfg)
        self.notifier  = Notifier(cfg=self.cfg)
        self._session_discoveries: list[str] = []
        logger.info("Acsis agent initialised — mission: understand the universe")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    async def think(self, question: str, notify: bool = True) -> ThinkResult:
        """
        Full Acsis loop on a question.

        Stages:
          1. RETRIEVE — what do we already know?
          2. VERIFY   — is it actually true? (web + source check)
          3. REASON   — what follows from verified knowledge?
          4. EXPERIMENT — can we test it numerically?
          5. DISCOVER — what is genuinely new here?
          6. GROW     — do we need new capacity? (SOMA hook)
        """
        logger.info(f"[THINK] Starting: {question[:80]}...")
        sources = []
        reasoning_chain = []
        discoveries = []
        experiments_run = 0
        knowledge_added = 0

        # ── STAGE 1: RETRIEVE ────────────────────────────────────────────────
        logger.info("[1/6] RETRIEVE — checking internal knowledge base")
        internal_hits = await self.memory.search(question, top_k=5)
        internal_context = self._format_hits(internal_hits)
        reasoning_chain.append(f"Retrieved {len(internal_hits)} relevant memories from knowledge base")

        # ── STAGE 2: VERIFY ──────────────────────────────────────────────────
        logger.info("[2/6] VERIFY — researching online")
        research_result = await self.research.investigate(
            question=question,
            context=internal_context,
            max_sources=self.cfg.max_sources_per_query,
        )
        sources.extend(research_result.sources)
        reasoning_chain.append(f"Verified against {len(research_result.sources)} external sources")
        reasoning_chain.append(f"Confidence after verification: {research_result.confidence:.0%}")

        if research_result.contradictions:
            reasoning_chain.append(f"⚠ Found {len(research_result.contradictions)} contradictions — flagging uncertainty")

        # ── STAGE 3: REASON ───────────────────────────────────────────────────
        logger.info("[3/6] REASON — applying inference")
        reason_result = await self.reasoner.reason(
            question=question,
            verified_facts=research_result.facts,
            context=internal_context,
        )
        reasoning_chain.extend(reason_result.steps)

        # ── STAGE 4: EXPERIMENT ───────────────────────────────────────────────
        if reason_result.requires_computation:
            logger.info("[4/6] EXPERIMENT — running code")
            for code_task in reason_result.code_tasks:
                exec_result = await self.executor.run(code_task)
                experiments_run += 1
                if exec_result.success:
                    reasoning_chain.append(f"Experiment {experiments_run}: {exec_result.interpretation}")
                    research_result.facts.append(exec_result.result_fact)
                else:
                    reasoning_chain.append(f"Experiment {experiments_run} failed: {exec_result.error}")
        else:
            logger.info("[4/6] EXPERIMENT — not required for this question")

        # ── STAGE 5: DISCOVER ─────────────────────────────────────────────────
        logger.info("[5/6] DISCOVER — checking for novel conclusions")
        novel = self._find_discoveries(
            reason_result=reason_result,
            existing_knowledge=internal_hits,
        )
        discoveries.extend(novel)
        self._session_discoveries.extend(novel)
        if novel:
            reasoning_chain.append(f"★ {len(novel)} potential discovery/discoveries flagged")

        # ── STAGE 6: GROW (SOMA hook) ─────────────────────────────────────────
        logger.info("[6/6] GROW — checking necessity")
        if reason_result.confidence < self.cfg.confidence_threshold and research_result.confidence < self.cfg.confidence_threshold:
            logger.info("  → Low confidence detected. SOMA-NECESSITY check triggered.")
            # In production this calls the SOMA system
            # soma_result = await self.soma.check_necessity(question, reason_result)
            reasoning_chain.append("Low confidence: flagged for SOMA adapter growth on next training cycle")

        # ── STORE EVERYTHING ──────────────────────────────────────────────────
        facts_to_store = research_result.facts + [reason_result.conclusion]
        for fact in facts_to_store:
            if fact and len(fact) > 10:
                await self.memory.store(fact, metadata={"source": "acsis_think", "question": question})
                await self.graph.add_fact(fact, question=question)
                knowledge_added += 1

        # ── BUILD RESULT ──────────────────────────────────────────────────────
        final_confidence = (research_result.confidence + reason_result.confidence) / 2
        result = ThinkResult(
            question=question,
            answer=reason_result.conclusion,
            confidence=final_confidence,
            sources=sources,
            reasoning_chain=reasoning_chain,
            discoveries=discoveries,
            experiments_run=experiments_run,
            knowledge_added=knowledge_added,
        )

        # ── NOTIFY ────────────────────────────────────────────────────────────
        if notify and self.cfg.notifications_enabled:
            await self.notifier.send(result.summary())

        # ── LOG ───────────────────────────────────────────────────────────────
        self._save_result(result)
        logger.info(f"[THINK] Complete. Confidence: {final_confidence:.0%}. Discoveries: {len(discoveries)}")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH MODE
    # ─────────────────────────────────────────────────────────────────────────

    async def investigate(self, questions: list[str]) -> list[ThinkResult]:
        """Run Acsis on multiple questions. Used for research campaigns."""
        results = []
        for i, q in enumerate(questions):
            logger.info(f"[INVESTIGATE] {i+1}/{len(questions)}: {q[:60]}...")
            result = await self.think(q, notify=(i == len(questions)-1))  # notify on last only
            results.append(result)
        await self.notifier.send(f"Research campaign complete: {len(results)} questions answered, {sum(len(r.discoveries) for r in results)} discoveries flagged.")
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # CONTINUOUS LOOP (run as a background process)
    # ─────────────────────────────────────────────────────────────────────────

    async def run_forever(self, question_queue_path: str):
        """
        Continuous research mode. Watches a file for new questions.
        Add a question to the file and Acsis picks it up.

        Format of questions.txt: one question per line.
        """
        logger.info(f"[RUN_FOREVER] Watching {question_queue_path}")
        processed: set[str] = set()
        while True:
            try:
                path = Path(question_queue_path)
                if path.exists():
                    with open(path) as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]
                    for q in lines:
                        if q not in processed:
                            logger.info(f"[RUN_FOREVER] New question: {q[:60]}")
                            await self.think(q)
                            processed.add(q)
                await asyncio.sleep(30)   # check every 30 seconds
            except KeyboardInterrupt:
                logger.info("[RUN_FOREVER] Stopping gracefully")
                break
            except Exception as e:
                logger.error(f"[RUN_FOREVER] Error: {e}")
                await asyncio.sleep(60)

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _format_hits(self, hits: list) -> str:
        if not hits:
            return "No relevant memories found."
        return "\n".join(f"- {h['text'][:200]}" for h in hits[:5])

    def _find_discoveries(self, reason_result, existing_knowledge: list) -> list[str]:
        """
        A conclusion is a discovery if:
        1. It appears in our reasoning output AND
        2. It was NOT in our existing knowledge base (semantic similarity < threshold)

        This is a heuristic. Full novelty detection requires Layer 3 formal verification.
        """
        discoveries = []
        existing_texts = {h['text'] for h in existing_knowledge}
        for candidate in reason_result.novel_claims:
            if not any(candidate.lower() in ex.lower() for ex in existing_texts):
                discoveries.append(candidate)
        return discoveries

    def _save_result(self, result: ThinkResult):
        path = Path(self.cfg.log_dir) / "results"
        path.mkdir(parents=True, exist_ok=True)
        fname = path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    import sys
    agent = AcsisAgent()
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Acsis > What do you want to understand? ")
    result = await agent.think(question)
    print(result.summary())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    asyncio.run(main())
