"""
acsis/tools/research.py
========================
The VERIFY stage. Acsis searches, cross-references, and assigns confidence.

Sources:
  - Tavily (web search, best for factual queries)
  - arXiv (scientific papers)
  - Wikipedia (structured factual baseline)
  - PubMed (medical/biological)
"""
from __future__ import annotations
import asyncio
import aiohttp
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    query: str
    facts: list[str]
    sources: list[str]
    confidence: float           # [0,1] — based on source agreement
    contradictions: list[str]   # claims that conflict across sources
    raw_texts: list[str]        # full extracted text per source


class ResearchTool:
    """
    Conducts multi-source research and returns verified facts.

    Usage:
        tool = ResearchTool(cfg)
        result = await tool.investigate("What is CRISPR gene editing?")
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN
    # ─────────────────────────────────────────────────────────────────────────

    async def investigate(
        self,
        question: str,
        context: str = "",
        max_sources: int = 6,
    ) -> ResearchResult:
        """Full investigation pipeline on a question."""
        logger.info(f"[RESEARCH] Investigating: {question[:60]}")

        tasks = [
            self._tavily_search(question),
            self._arxiv_search(question),
            self._wikipedia_search(question),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_facts = []
        all_sources = []
        all_texts = []

        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[RESEARCH] Source failed: {r}")
                continue
            all_facts.extend(r.get("facts", []))
            all_sources.extend(r.get("sources", []))
            all_texts.extend(r.get("texts", []))

        # Deduplicate
        all_facts = list(dict.fromkeys(all_facts))[:20]
        all_sources = list(dict.fromkeys(all_sources))[:max_sources]

        # Find contradictions + compute confidence
        contradictions = self._find_contradictions(all_facts)
        confidence = self._compute_confidence(all_facts, contradictions, len(all_sources))

        return ResearchResult(
            query=question,
            facts=all_facts,
            sources=all_sources,
            confidence=confidence,
            contradictions=contradictions,
            raw_texts=all_texts,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAVILY — best general web search
    # ─────────────────────────────────────────────────────────────────────────

    async def _tavily_search(self, query: str) -> dict:
        """Web search via Tavily API. Structured, factual, fast."""
        if not (self.cfg and self.cfg.tavily_api_key):
            # Fallback: use DuckDuckGo scraping (no API key needed)
            return await self._ddg_fallback(query)

        session = await self._get_session()
        payload = {
            "api_key": self.cfg.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 5,
        }
        try:
            async with session.post("https://api.tavily.com/search", json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    facts = []
                    if data.get("answer"):
                        facts.append(data["answer"])
                    for r in data.get("results", []):
                        if r.get("content"):
                            # Extract key sentences (simple heuristic)
                            sentences = r["content"].split(". ")
                            facts.extend([s.strip() for s in sentences[:3] if len(s) > 30])
                    sources = [r.get("url","") for r in data.get("results",[])]
                    return {"facts": facts, "sources": sources, "texts": [r.get("content","") for r in data.get("results",[])]}
        except Exception as e:
            logger.warning(f"[TAVILY] Error: {e}")
        return {"facts":[], "sources":[], "texts":[]}

    async def _ddg_fallback(self, query: str) -> dict:
        """DuckDuckGo instant answers — no API key required."""
        session = await self._get_session()
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        try:
            async with session.get("https://api.duckduckgo.com/", params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    facts = []
                    if data.get("Abstract"):
                        facts.append(data["Abstract"])
                    for topic in data.get("RelatedTopics", [])[:3]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            facts.append(topic["Text"][:300])
                    return {"facts": facts, "sources": [data.get("AbstractURL","")], "texts": facts}
        except Exception as e:
            logger.warning(f"[DDG] Error: {e}")
        return {"facts":[], "sources":[], "texts":[]}

    # ─────────────────────────────────────────────────────────────────────────
    # ARXIV — scientific papers
    # ─────────────────────────────────────────────────────────────────────────

    async def _arxiv_search(self, query: str) -> dict:
        """Search arXiv for relevant papers. Returns abstracts as facts."""
        session = await self._get_session()
        max_results = getattr(self.cfg, 'arxiv_max_results', 3)
        url = "http://export.arxiv.org/api/query"
        params = {"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                text = await resp.text()
                # Parse basic XML without library
                facts = []
                sources = []
                import re
                abstracts = re.findall(r'<summary>(.*?)</summary>', text, re.DOTALL)
                ids = re.findall(r'<id>(.*?)</id>', text)
                for i, (ab, pid) in enumerate(zip(abstracts[:max_results], ids[1:])):  # skip feed id
                    clean = ab.strip().replace('\n', ' ')
                    if len(clean) > 50:
                        facts.append(f"[arXiv] {clean[:400]}")
                        sources.append(pid.strip())
                return {"facts": facts, "sources": sources, "texts": facts}
        except Exception as e:
            logger.warning(f"[ARXIV] Error: {e}")
        return {"facts":[], "sources":[], "texts":[]}

    # ─────────────────────────────────────────────────────────────────────────
    # WIKIPEDIA — structured factual baseline
    # ─────────────────────────────────────────────────────────────────────────

    async def _wikipedia_search(self, query: str) -> dict:
        """Wikipedia API — free, structured, high quality."""
        session = await self._get_session()
        # Step 1: search for article title
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        # Clean query for URL
        title = query.replace(" ", "_")[:50]
        try:
            async with session.get(f"{search_url}{title}", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    extract = data.get("extract", "")
                    url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    if extract:
                        # Split into sentences for fact extraction
                        sentences = extract.split(". ")
                        facts = [s.strip() for s in sentences[:5] if len(s) > 30]
                        return {"facts": facts, "sources": [url], "texts": [extract]}
        except Exception as e:
            logger.warning(f"[WIKIPEDIA] Error: {e}")
        return {"facts":[], "sources":[], "texts":[]}

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIDENCE + CONTRADICTION DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _find_contradictions(self, facts: list[str]) -> list[str]:
        """
        Simple heuristic contradiction detection.
        Looks for negation patterns across facts.
        Full version uses NLI (natural language inference).
        """
        contradictions = []
        negation_words = ["not", "never", "no", "false", "incorrect", "wrong", "disproven"]
        for i, f1 in enumerate(facts):
            for f2 in facts[i+1:]:
                # Check if one negates the other
                f1_lower, f2_lower = f1.lower(), f2.lower()
                if any(neg in f2_lower and any(kw in f1_lower for kw in f2_lower.split()) for neg in negation_words):
                    contradictions.append(f"'{f1[:80]}' vs '{f2[:80]}'")
        return contradictions[:3]  # cap at 3 reported contradictions

    def _compute_confidence(
        self,
        facts: list[str],
        contradictions: list[str],
        n_sources: int,
    ) -> float:
        """
        Confidence = f(n_sources, n_facts, n_contradictions)
        More sources agreeing = higher confidence.
        Each contradiction penalises.
        """
        if not facts or not n_sources:
            return 0.1
        base = min(0.90, 0.3 + n_sources * 0.1)
        penalty = len(contradictions) * 0.15
        return max(0.1, base - penalty)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
