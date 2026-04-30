"""
soma/retrieval/search.py — Always-On Retrieval + Clarification
===============================================================
Always searches — even for things the model might already know.
Reason: world knowledge drifts, model has a cutoff, sources conflict.
We need to verify against current world knowledge at every step.

Clarification questions are generated when:
  - retrieved docs conflict with each other
  - curiosity mismatch direction suggests a specific gap
  - question is ambiguous (multiple valid interpretations)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class RetrieveAndClarify:
    """
    Always-on retrieval. Never skips search even for known topics.

    In paper 1: DuckDuckGo + arXiv + Wikipedia (all free, no API key)
    In paper 2: Tavily + PubMed + Semantic Scholar for richer retrieval
    """

    def __init__(self, always_search: bool = True, max_sources: int = 6):
        self.always_search = always_search
        self.max_sources = max_sources

    async def run(
        self,
        signal,                    # CuriositySignal
        context: str,
        force: bool = False,       # override is_learnable gate
    ) -> dict:
        """
        Retrieve world knowledge. Always runs if force=True.
        """
        if not self.always_search and not signal.is_learnable and not force:
            return {"retrieved_docs": [], "sources": [], "clarifying_questions": [], "skipped": True}

        logger.info(f"[RETRIEVE] Searching: {context[:60]}")

        results = await asyncio.gather(
            self._ddg(context),
            self._arxiv(context),
            self._wikipedia(context),
            return_exceptions=True,
        )

        docs, sources = [], []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[RETRIEVE] Source failed: {r}")
                continue
            docs.extend(r.get("docs", []))
            sources.extend(r.get("sources", []))

        docs    = list(dict.fromkeys(docs))[:self.max_sources * 3]
        sources = list(dict.fromkeys([s for s in sources if s]))[:self.max_sources]

        questions = self._generate_clarifications(signal, docs, context)

        return {
            "retrieved_docs": docs,
            "sources": sources,
            "clarifying_questions": questions,
            "skipped": False,
            "n_sources": len(sources),
        }

    async def _ddg(self, query: str) -> dict:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    data = await r.json(content_type=None)
            docs, sources = [], []
            if data.get("Abstract"):
                docs.append(data["Abstract"])
                sources.append(data.get("AbstractURL", ""))
            for t in data.get("RelatedTopics", [])[:3]:
                if isinstance(t, dict) and t.get("Text"):
                    docs.append(t["Text"][:300])
                    if t.get("FirstURL"): sources.append(t["FirstURL"])
            return {"docs": docs, "sources": sources}
        except Exception as e:
            logger.debug(f"[DDG] {e}")
            return {"docs": [], "sources": []}

    async def _arxiv(self, query: str) -> dict:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    "http://export.arxiv.org/api/query",
                    params={"search_query": f"all:{query}", "max_results": 3},
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as r:
                    text = await r.text()
            abstracts = re.findall(r"<summary>(.*?)</summary>", text, re.DOTALL)
            ids       = re.findall(r"<id>http.*?/abs/(.*?)</id>", text)
            docs    = [f"[arXiv] {a.strip()[:400]}" for a in abstracts[:3]]
            sources = [f"https://arxiv.org/abs/{pid}" for pid in ids[:3]]
            return {"docs": docs, "sources": sources}
        except Exception as e:
            logger.debug(f"[ARXIV] {e}")
            return {"docs": [], "sources": []}

    async def _wikipedia(self, query: str) -> dict:
        try:
            import aiohttp
            title = query.replace(" ", "_")[:50]
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as r:
                    if r.status != 200: return {"docs": [], "sources": []}
                    data = await r.json()
            extract = data.get("extract", "")
            url     = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            docs    = [s.strip() for s in extract.split(". ")[:5] if len(s) > 30]
            return {"docs": docs, "sources": [url]}
        except Exception as e:
            logger.debug(f"[WIKI] {e}")
            return {"docs": [], "sources": []}

    def _generate_clarifications(
        self, signal, docs: list, question: str
    ) -> list:
        """
        Generate clarifying questions when gaps are detected.

        Triggers:
          1. Retrieved docs conflict with each other (check for negations)
          2. Question is ambiguous (multiple "or" / "vs" patterns)
          3. Curiosity score is high but docs are sparse
        """
        questions = []

        # Ambiguity check
        if any(kw in question.lower() for kw in [" or ", " vs ", " versus ", "which one"]):
            questions.append(
                f"To clarify: are you asking about {question[:60]}? "
                "Should I compare all options or focus on one?"
            )

        # Sparse retrieval + high curiosity
        if len(docs) < 2 and signal.C > 0.6:
            questions.append(
                f"The question about '{question[:50]}' returned few sources. "
                "Can you provide more context or a specific domain?"
            )

        # Conflict detection (simple: one doc negates another)
        if len(docs) >= 2:
            neg_words = ["not", "never", "false", "incorrect", "disproven"]
            has_conflict = any(
                any(w in docs[j].lower() for w in neg_words)
                for j in range(min(3, len(docs)))
            )
            if has_conflict:
                questions.append(
                    "Sources contain conflicting information on this topic. "
                    "Should I prioritise recent sources or majority consensus?"
                )

        return questions[:2]  # cap at 2 clarifying questions
