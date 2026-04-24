"""
acsis/memory/vector_store.py
=============================
Long-term memory. Everything Acsis learns is stored here.
Uses ChromaDB (local, free, fast) with sentence-transformer embeddings.

This is the external "What do I know?" layer that persists across sessions.
"""
from __future__ import annotations
import logging
import uuid
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Persistent semantic memory using ChromaDB.

    Stores text facts as embeddings. Retrieves by semantic similarity.
    Survives model retraining — external knowledge is never lost.

    Usage:
        store = VectorStore()
        await store.store("Malaria is caused by the plasmodium parasite")
        results = await store.search("What causes malaria?")
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.path = getattr(cfg, 'chroma_path', './acsis_chroma') if cfg else './acsis_chroma'
        self._client = None
        self._collection = None
        self._embedder = None

    def _init(self):
        """Lazy init — only load when first used."""
        if self._client is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(path=self.path)
            self._collection = self._client.get_or_create_collection(
                name="acsis_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"[MEMORY] ChromaDB loaded: {self._collection.count()} facts stored")
        except ImportError:
            logger.warning("[MEMORY] ChromaDB not installed. Using in-memory fallback.")
            self._collection = InMemoryStore()

        try:
            from sentence_transformers import SentenceTransformer
            model_name = getattr(self.cfg, 'embedding_model', 'all-MiniLM-L6-v2') if self.cfg else 'all-MiniLM-L6-v2'
            self._embedder = SentenceTransformer(model_name)
            logger.info(f"[MEMORY] Embedder loaded: {model_name}")
        except ImportError:
            logger.warning("[MEMORY] sentence-transformers not installed. Using hash-based fallback.")

    def _embed(self, text: str) -> list[float]:
        """Embed text to a vector."""
        if self._embedder:
            return self._embedder.encode(text, normalize_embeddings=True).tolist()
        # Hash-based fallback (not semantic, just for testing)
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [b/255.0 for b in h] * 24  # 384-dim placeholder

    async def store(self, text: str, metadata: Optional[dict] = None) -> str:
        """
        Store a fact in the knowledge base.
        Returns the ID of the stored fact.
        """
        self._init()
        if not text or len(text.strip()) < 10:
            return ""

        doc_id = str(uuid.uuid4())
        meta = {
            "timestamp": datetime.now().isoformat(),
            "source": "acsis",
            **(metadata or {}),
        }

        try:
            embedding = self._embed(text)
            self._collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[meta],
            )
            logger.debug(f"[MEMORY] Stored: {text[:60]}...")
        except Exception as e:
            logger.error(f"[MEMORY] Store failed: {e}")

        return doc_id

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search over stored knowledge.
        Returns top_k most relevant facts with their metadata.
        """
        self._init()
        try:
            embedding = self._embed(query)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, max(1, self._collection.count())),
            )
            hits = []
            if results and results.get("documents"):
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    hits.append({
                        "text": doc,
                        "metadata": meta,
                        "similarity": 1 - dist,   # cosine: distance → similarity
                    })
            return hits
        except Exception as e:
            logger.warning(f"[MEMORY] Search failed: {e}")
            return []

    async def count(self) -> int:
        """Total facts stored."""
        self._init()
        try:
            return self._collection.count()
        except Exception:
            return 0

    async def delete(self, doc_id: str):
        """Remove a specific fact (for knowledge correction)."""
        self._init()
        try:
            self._collection.delete(ids=[doc_id])
        except Exception as e:
            logger.warning(f"[MEMORY] Delete failed: {e}")


class InMemoryStore:
    """Fallback when ChromaDB isn't installed. Not persistent."""
    def __init__(self):
        self._docs = []
        self._metas = []

    def count(self):
        return len(self._docs)

    def add(self, documents, embeddings, ids, metadatas):
        self._docs.extend(documents)
        self._metas.extend(metadatas)

    def query(self, query_embeddings, n_results):
        # Return last n_results (no actual semantic search)
        n = min(n_results, len(self._docs))
        return {
            "documents": [self._docs[-n:]],
            "metadatas": [self._metas[-n:]],
            "distances": [[0.1] * n],
        }

    def delete(self, ids):
        pass
