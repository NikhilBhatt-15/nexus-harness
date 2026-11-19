"""nexus-lite: an open, local knowledge engine modeled on Pinecone Nexus.

Pipeline:  sources -> curate() -> Context (artifacts + hybrid index) -> query()

- ``corpus``    load source documents (markdown/JSON) and group them by entity
- ``chunking``  heading-aware chunker (target/overlap in tokens, approximated)
- ``embed``     pluggable embedders (local Ollama by default; free)
- ``index``     BM25 + dense hybrid retrieval with reciprocal-rank fusion
- ``compiler``  curate(): deterministic extractors + optional LLM manifest extraction
- ``context``   on-disk compiled Context (SQLite artifacts + chunks + vectors)
- ``query``     query(): KnowQL-lite request -> routed, composed, cited answer
- ``server``    HTTP API compatible with the Nexus v0 query surface
"""

__version__ = "0.1.0"
