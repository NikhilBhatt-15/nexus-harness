"""Hybrid retrieval: BM25 (sparse) + embeddings (dense), fused with RRF."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from nexus_lite.embed import Embedder

_TOKEN = re.compile(r"[a-z0-9$%.]+(?:_[a-z0-9]+)*")
_STOP = set(
    "a an and are as at be by can do does for from has have how i if in is it its of on or "
    "that the their this to was what when where which who will with you your my me we our".split()
)


def tokenize(text: str) -> list[str]:
    toks = [t.strip(".") for t in _TOKEN.findall(text.lower())]
    return [t for t in toks if t and t not in _STOP]


class HybridIndex:
    def __init__(self, ids: list[str], texts: list[str], vectors: np.ndarray | None, embedder: Embedder | None):
        self.ids = ids
        self.texts = texts
        self.vectors = vectors
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    def search_bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def search_dense(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.vectors is None or self.embedder is None or not len(self.vectors):
            return []
        q = self.embedder.embed([query], is_query=True)[0]
        sims = self.vectors @ q
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def search(self, query: str, k: int = 8, *, rrf_k: int = 60, pool: int = 40) -> list[tuple[str, float]]:
        fused: dict[int, float] = {}
        for ranked in (self.search_bm25(query, pool), self.search_dense(query, pool)):
            for rank, (i, _) in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(self.ids[i], s) for i, s in best]
