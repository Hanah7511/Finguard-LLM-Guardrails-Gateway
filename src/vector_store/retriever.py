"""Regulation retrieval: citation lookup first, then FAISS similarity search.

FAISS is optional; without it the same inner-product search runs in NumPy (fine for
corpora of a few thousand chunks). The embedding model loads on first use.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegulationDocument:
    doc_id: str
    citation: str
    title: str
    text: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dim)."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype="float32")


def normalize_citation(citation: str) -> str:
    """'BSA 31 U.S.C. § 5324(a)' -> '31usc5324a'; drops act prefixes and punctuation."""
    text = citation.lower().replace("§", " ")
    text = re.sub(r"\b(bsa|bank secrecy act|section|sec)\b", " ", text)
    return re.sub(r"[^a-z0-9]", "", text)


class RegulationRetriever:
    def __init__(self, documents: list[RegulationDocument], embedder: Optional[Embedder] = None) -> None:
        if not documents:
            raise ValueError("RegulationRetriever needs at least one document")
        self.documents = documents
        self._embedder = embedder
        self._index = None
        self._matrix: Optional[np.ndarray] = None
        self._citation_keys = [
            (normalize_citation(c), doc) for doc in documents for c in (doc.citation, *doc.aliases)
        ]

    @classmethod
    def from_json(cls, path: Path, embedder: Optional[Embedder] = None) -> "RegulationRetriever":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = [
            RegulationDocument(
                doc_id=d["doc_id"],
                citation=d["citation"],
                title=d["title"],
                text=d["text"],
                aliases=tuple(d.get("aliases", [])),
            )
            for d in raw["documents"]
        ]
        return cls(docs, embedder)

    def find_by_citation(self, citation: str, min_similarity: float = 0.55) -> Optional[RegulationDocument]:
        """Exact citation match, else the nearest document if it is close enough."""
        key = normalize_citation(citation)
        if key:
            # "31usc5324a" contains "31usc5324". Longest key first so the most
            # specific document wins. A cited key shorter than the document's
            # (e.g. "31usc53") never matches: that would accept vague citations.
            for doc_key, doc in sorted(self._citation_keys, key=lambda kd: -len(kd[0])):
                if doc_key and doc_key in key:
                    return doc
        hits = self.search(citation, k=1)
        if hits and hits[0][1] >= min_similarity:
            return hits[0][0]
        return None

    def search(self, query: str, k: int = 3) -> list[tuple[RegulationDocument, float]]:
        k = min(k, len(self.documents))
        if self._embedder is None:
            return self._lexical_search(query, k)
        self._ensure_index()
        q = self._embedder.encode([query])
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            pairs = zip(idx[0].tolist(), scores[0].tolist())
        else:
            sims = (self._matrix @ q[0]).astype(float)
            order = np.argsort(-sims)[:k]
            pairs = ((int(i), float(sims[i])) for i in order)
        return [(self.documents[i], round(float(s), 4)) for i, s in pairs if i >= 0]

    def _ensure_index(self) -> None:
        if self._matrix is not None:
            return
        texts = [f"{d.citation}. {d.title}. {d.text}" for d in self.documents]
        self._matrix = self._embedder.encode(texts)
        try:
            import faiss

            index = faiss.IndexFlatIP(self._matrix.shape[1])
            index.add(self._matrix)
            self._index = index
        except ImportError:
            logger.info("faiss not installed; using NumPy similarity search")

    def _lexical_search(self, query: str, k: int) -> list[tuple[RegulationDocument, float]]:
        """Token-overlap fallback when no embedder is configured."""
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for doc in self.documents:
            d = set(re.findall(r"[a-z0-9]+", f"{doc.citation} {doc.title} {doc.text}".lower()))
            scored.append((doc, len(q & d) / len(q | d) if q else 0.0))
        scored.sort(key=lambda p: -p[1])
        return [(doc, round(s, 4)) for doc, s in scored[:k]]
