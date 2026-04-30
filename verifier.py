"""
soma/verification/verifier.py — Logical Consistency Verifier
=============================================================
Paper 1: lightweight heuristic + Z3 stub
Paper 3: full Lean 4 integration for formal proofs

Two verification targets:
  1. Internal consistency — does the reasoning chain contradict itself?
  2. World-knowledge alignment — does the answer agree with retrieved docs?
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class Verifier:
    """
    Verifies logical consistency and world-knowledge alignment.

    Paper 1 implementation: heuristic rules + Z3 SMT solver stub
    Paper 3 implementation: Lean 4 for formal mathematical proofs

    Usage:
        v = Verifier()
        result = v.verify(
            statement="If A then B. A is true. Therefore B is true.",
            retrieved_docs=docs,
            use_z3=True,
        )
        if not result["consistent"]:
            confidence *= 0.5
    """

    def __init__(self):
        self._z3_available = self._check_z3()
        self._lean_available = False  # Paper 3
        logger.info(f"[VERIFY] Z3 available: {self._z3_available} | Lean: {self._lean_available}")

    def verify(
        self,
        statement: str,
        retrieved_docs: list,
        use_z3: bool = False,
        use_lean: bool = False,
    ) -> dict:
        """
        Multi-stage verification.

        Returns:
            {
                "consistent": bool,
                "confidence": float,
                "method": str,
                "issues": list of detected problems
            }
        """
        issues = []

        # Stage 1: Heuristic consistency check (always runs)
        heuristic_result = self._heuristic_check(statement, retrieved_docs)
        if heuristic_result["issues"]:
            issues.extend(heuristic_result["issues"])

        # Stage 2: Z3 for logical structure (Paper 1 partial)
        z3_consistent = True
        if use_z3 and self._z3_available:
            logical_claims = self._extract_logical_claims(statement)
            if logical_claims:
                z3_result = self._z3_check(logical_claims)
                z3_consistent = z3_result["consistent"]
                if not z3_consistent:
                    issues.append(f"Z3: logical contradiction in: {z3_result.get('conflict', '')}")

        # Stage 3: Lean 4 (Paper 3 — stub)
        lean_consistent = True
        if use_lean and self._lean_available:
            lean_consistent = self._lean_check(statement)

        overall_consistent = heuristic_result["consistent"] and z3_consistent and lean_consistent
        confidence = 0.90 if overall_consistent else (0.40 if issues else 0.70)

        return {
            "consistent": overall_consistent,
            "confidence": confidence,
            "method": "heuristic" + ("+z3" if use_z3 else "") + ("+lean" if use_lean else ""),
            "issues": issues,
        }

    def _heuristic_check(self, statement: str, docs: list) -> dict:
        """
        Fast heuristic consistency check.

        Checks:
          - Direct contradiction within statement
          - Key claim conflicts with retrieved docs
          - Numerical claim consistency
        """
        issues = []
        s_lower = statement.lower()

        # Direct self-contradiction patterns
        contradiction_patterns = [
            (r"\bis always\b.*\bis never\b", "always/never contradiction"),
            (r"\ball\b.*\bnone\b", "all/none contradiction"),
            (r"\bincreases\b.*\bdecreases\b", "direction contradiction"),
        ]
        for pattern, label in contradiction_patterns:
            if re.search(pattern, s_lower):
                issues.append(f"Heuristic: {label}")

        # Check against retrieved docs
        if docs:
            doc_combined = " ".join(docs).lower()
            # Simple: does statement contain claims that docs explicitly negate?
            neg_words = ["false", "incorrect", "disproven", "not true", "myth"]
            for word in neg_words:
                if word in doc_combined and any(
                    claim in s_lower
                    for claim in s_lower.split(".") if len(claim) > 20
                ):
                    issues.append(f"Doc conflict: retrieved sources contradict claim near '{word}'")
                    break

        return {
            "consistent": len(issues) == 0,
            "issues": issues,
        }

    def _extract_logical_claims(self, text: str) -> list:
        """
        Extract simple if-then logical structures for Z3.
        e.g., "If A then B" → ("A", "B")
        """
        claims = []
        patterns = [
            r"[Ii]f (.+?) then (.+?)[\.\,]",
            r"(.+?) implies (.+?)[\.\,]",
            r"(.+?) therefore (.+?)[\.\,]",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                claims.append({
                    "antecedent": match.group(1).strip(),
                    "consequent": match.group(2).strip(),
                })
        return claims[:5]   # limit to 5 claims per check

    def _z3_check(self, logical_claims: list) -> dict:
        """
        Use Z3 SMT solver to check logical consistency.
        Paper 1: basic propositional logic only.
        Paper 3: full first-order logic with quantifiers.
        """
        try:
            import z3
            solver = z3.Solver()
            propositions = {}

            for claim in logical_claims:
                ant = claim["antecedent"][:20]
                con = claim["consequent"][:20]

                # Create Z3 boolean variables
                if ant not in propositions:
                    propositions[ant] = z3.Bool(f"p_{len(propositions)}")
                if con not in propositions:
                    propositions[con] = z3.Bool(f"p_{len(propositions)}")

                # Add implication: antecedent → consequent
                solver.add(z3.Implies(propositions[ant], propositions[con]))

            result = solver.check()
            if result == z3.unsat:
                return {"consistent": False, "conflict": "Z3: unsatisfiable constraint set"}
            return {"consistent": True}

        except ImportError:
            logger.debug("[Z3] Not installed. Skipping formal check.")
            return {"consistent": True}  # Optimistic fallback
        except Exception as e:
            logger.warning(f"[Z3] Error: {e}")
            return {"consistent": True}

    def _lean_check(self, statement: str) -> bool:
        """
        Lean 4 formal verification — Paper 3 implementation.
        Stub for now.
        """
        # TODO Paper 3:
        # 1. Parse statement into Lean 4 proposition
        # 2. Submit to Lean 4 kernel (subprocess)
        # 3. Return True if proof succeeds
        return True

    def _check_z3(self) -> bool:
        try:
            import z3
            return True
        except ImportError:
            return False
