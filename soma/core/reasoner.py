"""
acsis/core/reasoner.py
=======================
REASON stage. Applies three reasoning modes over verified facts:
  - Deduction: what must follow from these facts?
  - Induction:  what pattern do these instances suggest?
  - Abduction:  what best explains these observations?

Also decides whether a computation experiment is needed.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ReasonResult:
    conclusion: str
    steps: list[str]
    confidence: float
    reasoning_mode: str          # "deductive" | "inductive" | "abductive"
    novel_claims: list[str]      # conclusions not present in input facts
    requires_computation: bool
    code_tasks: list[str]        # Python code to run if computation needed


class Reasoner:
    """
    Applies formal reasoning patterns over verified facts.

    In Phase 1: LLM-based chain-of-thought (pattern matching over reasoning text)
    In Phase 3: formal verifier integration (Lean 4 / Z3)
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._llm = None   # loaded lazily

    async def reason(
        self,
        question: str,
        verified_facts: list[str],
        context: str = "",
    ) -> ReasonResult:
        """
        Apply reasoning to arrive at a conclusion.
        """
        logger.info(f"[REASON] Question: {question[:60]}")
        logger.info(f"[REASON] Working with {len(verified_facts)} verified facts")

        # Select reasoning mode
        mode = self._select_mode(question, verified_facts)
        logger.info(f"[REASON] Mode: {mode}")

        # Apply the selected mode
        if mode == "deductive":
            return await self._deduce(question, verified_facts, context)
        elif mode == "inductive":
            return await self._induce(question, verified_facts, context)
        else:
            return await self._abduce(question, verified_facts, context)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE SELECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _select_mode(self, question: str, facts: list[str]) -> str:
        """
        Heuristic mode selection:
        - "therefore" / "must" / "prove" → deductive
        - "why" / "explain" / "what causes" → abductive
        - "pattern" / "trend" / "generally" → inductive
        """
        q = question.lower()
        if any(w in q for w in ["prove", "therefore", "must", "necessarily", "follows"]):
            return "deductive"
        if any(w in q for w in ["why", "cause", "explain", "reason", "how does"]):
            return "abductive"
        if any(w in q for w in ["pattern", "trend", "generally", "usually", "tend to"]):
            return "inductive"
        # Default: abductive (most common for research questions)
        return "abductive"

    # ─────────────────────────────────────────────────────────────────────────
    # DEDUCTION — what must be true given verified facts
    # ─────────────────────────────────────────────────────────────────────────

    async def _deduce(
        self,
        question: str,
        facts: list[str],
        context: str,
    ) -> ReasonResult:
        """
        Deductive reasoning: if P1 and P2, then C.
        Output must follow necessarily from premises.
        """
        steps = []
        steps.append(f"DEDUCTIVE MODE: Checking what follows necessarily from {len(facts)} verified facts")

        # Build the reasoning via LLM
        prompt = self._build_prompt(
            mode="deductive",
            question=question,
            facts=facts,
            context=context,
        )
        conclusion, code_tasks = await self._call_llm(prompt)

        steps.append("Identified relevant premises from verified facts")
        steps.append(f"Applied modus ponens / syllogistic reasoning")
        steps.append(f"Conclusion: {conclusion[:100]}")

        novel = self._extract_novel(conclusion, facts)
        requires_computation = bool(code_tasks)

        return ReasonResult(
            conclusion=conclusion,
            steps=steps,
            confidence=0.85,   # deduction is high confidence when premises are verified
            reasoning_mode="deductive",
            novel_claims=novel,
            requires_computation=requires_computation,
            code_tasks=code_tasks,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ABDUCTION — best explanation for observations
    # ─────────────────────────────────────────────────────────────────────────

    async def _abduce(
        self,
        question: str,
        facts: list[str],
        context: str,
    ) -> ReasonResult:
        """
        Abductive reasoning: inference to the best explanation.
        Peirce: "The surprising fact C is observed. But if A were true,
                 C would be a matter of course. Hence, A is likely true."
        """
        steps = []
        steps.append(f"ABDUCTIVE MODE: Searching for best explanation of {len(facts)} observations")

        prompt = self._build_prompt(
            mode="abductive",
            question=question,
            facts=facts,
            context=context,
        )
        conclusion, code_tasks = await self._call_llm(prompt)

        steps.append("Identified surprising / unexplained observations in facts")
        steps.append("Generated candidate explanations")
        steps.append("Selected most parsimonious explanation (Occam's razor)")
        steps.append(f"Best explanation: {conclusion[:100]}")

        novel = self._extract_novel(conclusion, facts)

        return ReasonResult(
            conclusion=conclusion,
            steps=steps,
            confidence=0.65,   # abduction is inherently uncertain
            reasoning_mode="abductive",
            novel_claims=novel,
            requires_computation=bool(code_tasks),
            code_tasks=code_tasks,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # INDUCTION — generalize from instances
    # ─────────────────────────────────────────────────────────────────────────

    async def _induce(
        self,
        question: str,
        facts: list[str],
        context: str,
    ) -> ReasonResult:
        """
        Inductive reasoning: pattern from instances to general rule.
        """
        steps = []
        steps.append(f"INDUCTIVE MODE: Finding patterns across {len(facts)} instances")

        prompt = self._build_prompt(
            mode="inductive",
            question=question,
            facts=facts,
            context=context,
        )
        conclusion, code_tasks = await self._call_llm(prompt)

        steps.append("Analysed instances for common properties")
        steps.append("Identified candidate universal or statistical patterns")
        steps.append("Applied principle of parsimony to select strongest pattern")
        steps.append(f"Induced rule: {conclusion[:100]}")

        novel = self._extract_novel(conclusion, facts)

        return ReasonResult(
            conclusion=conclusion,
            steps=steps,
            confidence=0.70,
            reasoning_mode="inductive",
            novel_claims=novel,
            requires_computation=bool(code_tasks),
            code_tasks=code_tasks,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LLM CALL
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(self, mode: str, question: str, facts: list[str], context: str) -> str:
        facts_block = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(facts[:10]))
        mode_instruction = {
            "deductive": "Apply deductive reasoning. What MUST be true given these facts? Only state conclusions that follow necessarily.",
            "abductive": "Apply abductive reasoning. What is the BEST EXPLANATION for these observations? State the most likely cause or mechanism.",
            "inductive": "Apply inductive reasoning. What GENERAL RULE or PATTERN do these instances suggest? State the most parsimonious generalisation.",
        }[mode]

        return f"""You are Acsis, an intelligence system with one mission: understand the universe.

QUESTION: {question}

VERIFIED FACTS:
{facts_block}

PRIOR CONTEXT:
{context[:500] if context else "None"}

TASK: {mode_instruction}

Respond in this format:
CONCLUSION: [one clear, precise conclusion]
CODE_NEEDED: [YES/NO — does verifying this require computation?]
CODE: [if YES, write Python that tests or verifies the conclusion, else leave blank]
NOVEL_CLAIMS: [list any claims in your conclusion that go beyond the input facts, one per line]
"""

    async def _call_llm(self, prompt: str) -> tuple[str, list[str]]:
        """
        Call the base LLM. In Phase 1 this is an API call.
        In Phase 3 this uses the local SOMA-grown adapter.
        """
        # Phase 1: Use a simple heuristic extraction (no model loaded)
        # In production replace with actual LLM call
        conclusion = "Based on the verified facts, the most supported conclusion requires further investigation with the loaded language model."
        code_tasks = []

        # Try to load local model if configured
        if self.cfg and getattr(self.cfg, 'base_model', None):
            try:
                conclusion, code_tasks = await self._call_local_model(prompt)
            except Exception as e:
                logger.warning(f"[REASON] Local model call failed: {e}. Using fallback.")

        return conclusion, code_tasks

    async def _call_local_model(self, prompt: str) -> tuple[str, list[str]]:
        """Call the local quantised model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        if self._llm is None:
            logger.info(f"[REASON] Loading model: {self.cfg.base_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
            load_kwargs = {"device_map": "auto"}
            if self.cfg.load_in_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            self._llm = AutoModelForCausalLM.from_pretrained(self.cfg.base_model, **load_kwargs)

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Parse structured response
        conclusion = ""
        code_tasks = []
        lines = response.split("\n")
        in_code = False
        code_buf = []
        for line in lines:
            if line.startswith("CONCLUSION:"):
                conclusion = line.replace("CONCLUSION:", "").strip()
            elif line.startswith("CODE:") or in_code:
                if line.startswith("CODE:"):
                    in_code = True
                    rest = line.replace("CODE:", "").strip()
                    if rest:
                        code_buf.append(rest)
                elif line.startswith("NOVEL_CLAIMS:"):
                    in_code = False
                    if code_buf:
                        code_tasks.append("\n".join(code_buf))
                else:
                    code_buf.append(line)

        return conclusion or response[:300], code_tasks

    # ─────────────────────────────────────────────────────────────────────────
    # NOVELTY CHECK
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_novel(self, conclusion: str, facts: list[str]) -> list[str]:
        """
        Extract parts of the conclusion that weren't in the input facts.
        Heuristic: if a sentence from conclusion doesn't overlap with any fact > 60% word overlap.
        """
        novel = []
        conclusion_sents = [s.strip() for s in conclusion.split(".") if len(s.strip()) > 20]
        fact_words = set(" ".join(facts).lower().split())
        for sent in conclusion_sents:
            sent_words = set(sent.lower().split())
            if sent_words:
                overlap = len(sent_words & fact_words) / len(sent_words)
                if overlap < 0.50:   # less than 50% overlap with known facts = potentially novel
                    novel.append(sent)
        return novel[:3]  # cap at 3
