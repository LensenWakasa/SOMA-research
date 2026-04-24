"""
acsis/memory/knowledge_graph.py
================================
Structured knowledge — stores relationships between concepts.
Uses Neo4j when available, falls back to a simple in-memory graph.

Every fact Acsis learns gets stored as nodes + edges:
  "Malaria" → CAUSED_BY → "Plasmodium parasite"
  "CRISPR" → IS_TYPE_OF → "Gene editing"
  "SOMA" → ACHIEVES → "Continual learning without forgetting"
"""
from __future__ import annotations
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Stores structured relationships between concepts.

    Usage:
        kg = KnowledgeGraph()
        await kg.add_fact("Malaria is caused by the plasmodium parasite")
        neighbours = await kg.get_related("malaria")
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._driver = None
        self._fallback: dict = {"nodes": {}, "edges": []}
        self._use_neo4j = False
        self._try_connect()

    def _try_connect(self):
        """Attempt Neo4j connection. Fall back gracefully."""
        if not self.cfg:
            return
        try:
            from neo4j import GraphDatabase
            uri = getattr(self.cfg, 'neo4j_uri', 'bolt://localhost:7687')
            user = getattr(self.cfg, 'neo4j_user', 'neo4j')
            pwd = getattr(self.cfg, 'neo4j_password', 'acsis2026')
            self._driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self._driver.verify_connectivity()
            self._use_neo4j = True
            logger.info("[GRAPH] Neo4j connected")
        except Exception:
            logger.info("[GRAPH] Neo4j not available — using in-memory graph")
            self._use_neo4j = False

    async def add_fact(self, fact: str, question: str = "") -> bool:
        """
        Parse a fact string into (subject, relation, object) triples
        and store them in the graph.
        """
        triples = self._extract_triples(fact)
        for subj, rel, obj in triples:
            if self._use_neo4j and self._driver:
                self._neo4j_add(subj, rel, obj)
            else:
                self._memory_add(subj, rel, obj)
        return len(triples) > 0

    async def get_related(self, concept: str, depth: int = 2) -> list[dict]:
        """
        Get all nodes related to a concept up to depth hops.
        Returns list of {node, relation, target} dicts.
        """
        if self._use_neo4j and self._driver:
            return self._neo4j_query(concept, depth)
        return self._memory_query(concept)

    async def find_connections(self, concept_a: str, concept_b: str) -> list[str]:
        """
        Find if two concepts are connected, and how.
        Used for discovery — does A relate to B in any way?
        """
        if self._use_neo4j and self._driver:
            return self._neo4j_path(concept_a, concept_b)
        # In-memory: simple BFS
        return self._memory_path(concept_a, concept_b)

    # ─────────────────────────────────────────────────────────────────────────
    # TRIPLE EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_triples(self, text: str) -> list[tuple[str,str,str]]:
        """
        Extract (subject, relation, object) triples from natural language.
        Simple rule-based extraction. Phase 3 will use an NLP model.

        Examples:
          "Malaria is caused by the plasmodium parasite"
           → ("malaria", "IS_CAUSED_BY", "plasmodium parasite")
          "CRISPR is a type of gene editing"
           → ("crispr", "IS_TYPE_OF", "gene editing")
        """
        triples = []
        text = text.strip()
        patterns = [
            (r"(.+?) is caused by (.+)", "IS_CAUSED_BY"),
            (r"(.+?) is a type of (.+)", "IS_TYPE_OF"),
            (r"(.+?) was discovered by (.+)", "DISCOVERED_BY"),
            (r"(.+?) is part of (.+)", "IS_PART_OF"),
            (r"(.+?) leads to (.+)", "LEADS_TO"),
            (r"(.+?) prevents (.+)", "PREVENTS"),
            (r"(.+?) causes (.+)", "CAUSES"),
            (r"(.+?) treats (.+)", "TREATS"),
            (r"(.+?) is related to (.+)", "RELATED_TO"),
            (r"(.+?) achieves (.+)", "ACHIEVES"),
            (r"(.+?) enables (.+)", "ENABLES"),
        ]
        for pattern, relation in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                subj = match.group(1).strip().lower()[:50]
                obj  = match.group(2).strip().lower()[:50]
                # Clean punctuation
                obj = obj.rstrip(".,;")
                if len(subj) > 1 and len(obj) > 1:
                    triples.append((subj, relation, obj))
        return triples

    # ─────────────────────────────────────────────────────────────────────────
    # NEO4J OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _neo4j_add(self, subj: str, rel: str, obj: str):
        with self._driver.session() as session:
            session.run(
                f"MERGE (a:Concept {{name: $subj}}) "
                f"MERGE (b:Concept {{name: $obj}}) "
                f"MERGE (a)-[:{rel}]->(b)",
                subj=subj, obj=obj
            )

    def _neo4j_query(self, concept: str, depth: int) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (a:Concept {{name: $name}})-[r*1..{depth}]-(b) "
                f"RETURN type(r[0]) as relation, b.name as target LIMIT 20",
                name=concept.lower()
            )
            return [{"node": concept, "relation": rec["relation"], "target": rec["target"]} for rec in result]

    def _neo4j_path(self, a: str, b: str) -> list[str]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH p = shortestPath((a:Concept {name: $a})-[*]-(b:Concept {name: $b})) "
                "RETURN [n in nodes(p) | n.name] as path",
                a=a.lower(), b=b.lower()
            )
            record = result.single()
            return record["path"] if record else []

    # ─────────────────────────────────────────────────────────────────────────
    # IN-MEMORY FALLBACK
    # ─────────────────────────────────────────────────────────────────────────

    def _memory_add(self, subj: str, rel: str, obj: str):
        if subj not in self._fallback["nodes"]:
            self._fallback["nodes"][subj] = {"name": subj, "edges": []}
        self._fallback["nodes"][subj]["edges"].append({"relation": rel, "target": obj})
        self._fallback["edges"].append({"from": subj, "relation": rel, "to": obj})

    def _memory_query(self, concept: str) -> list[dict]:
        concept = concept.lower()
        results = []
        for edge in self._fallback["edges"]:
            if edge["from"] == concept or edge["to"] == concept:
                results.append(edge)
        return results[:20]

    def _memory_path(self, a: str, b: str) -> list[str]:
        """BFS to find path between two concepts."""
        a, b = a.lower(), b.lower()
        from collections import deque
        queue = deque([[a]])
        visited = {a}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == b:
                return path
            node_data = self._fallback["nodes"].get(node, {})
            for edge in node_data.get("edges", []):
                target = edge["target"]
                if target not in visited:
                    visited.add(target)
                    queue.append(path + [target])
        return []

    async def stats(self) -> dict:
        """Return graph statistics."""
        if self._use_neo4j and self._driver:
            with self._driver.session() as s:
                nodes = s.run("MATCH (n) RETURN count(n) as c").single()["c"]
                edges = s.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
                return {"nodes": nodes, "edges": edges, "backend": "neo4j"}
        return {
            "nodes": len(self._fallback["nodes"]),
            "edges": len(self._fallback["edges"]),
            "backend": "in-memory",
        }
