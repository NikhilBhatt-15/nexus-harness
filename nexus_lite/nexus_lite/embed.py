"""Pluggable embedders.

Spec strings:
  ``ollama:<model>``   local Ollama (default ``ollama:embeddinggemma-2``; free)
  ``litellm:<model>``  any litellm embedding model (e.g. ``litellm:text-embedding-3-large``)
  ``none``             disable dense retrieval (BM25 only)
"""

from __future__ import annotations

import os

import httpx
import numpy as np

DEFAULT_EMBEDDER = os.environ.get("NEXUS_LITE_EMBEDDER", "ollama:embeddinggemma-2")


class Embedder:
    spec: str

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    def __init__(self, model: str, url: str | None = None, batch: int = 32):
        self.spec = f"ollama:{model}"
        self.model = model
        self.url = (url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.url.startswith("http"):
            self.url = "http://" + self.url
        self.batch = batch

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        # embeddinggemma uses task prefixes; harmless for other models.
        prefix = "task: search result | query: " if is_query else "title: none | text: "
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            batch = [prefix + t for t in texts[i : i + self.batch]]
            r = httpx.post(f"{self.url}/api/embed", json={"model": self.model, "input": batch}, timeout=600)
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return _normalize(np.asarray(out, dtype=np.float32))


class LiteLLMEmbedder(Embedder):
    def __init__(self, model: str, batch: int = 64):
        self.spec = f"litellm:{model}"
        self.model = model
        self.batch = batch

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        import litellm

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = litellm.embedding(model=self.model, input=texts[i : i + self.batch])
            out.extend(d["embedding"] for d in resp.data)
        return _normalize(np.asarray(out, dtype=np.float32))


def _normalize(m: np.ndarray) -> np.ndarray:
    if m.size == 0:
        return m
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def get_embedder(spec: str | None = None) -> Embedder | None:
    spec = spec or DEFAULT_EMBEDDER
    if spec in ("", "none"):
        return None
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return OllamaEmbedder(model)
    if kind == "litellm":
        return LiteLLMEmbedder(model)
    raise ValueError(f"unknown embedder spec {spec!r}")
