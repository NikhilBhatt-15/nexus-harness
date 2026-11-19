"""A compiled Context on disk.

Layout of ``<root>/<slug>/``::

    context.sqlite   documents, chunks, extracted tables, artifacts
    vectors.npy      dense vectors aligned with chunk_ids.json (absent if no embedder)
    chunk_ids.json
    meta.json        slug, manifest, embedder, chunking, build stats

``compile_context`` is incremental (content-hash keyed).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from nexus_lite.chunking import chunk_document
from nexus_lite.compiler import PER_DOC_TABLES, SCHEMA, build_corpus_artifacts, extract_doc, llm_extract
from nexus_lite.corpus import Document
from nexus_lite.embed import Embedder, get_embedder
from nexus_lite.index import HybridIndex


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


def compile_context(
    root: str | Path,
    slug: str,
    docs: list[Document],
    *,
    embedder_spec: str | None = None,
    manifest: dict | None = None,
    llm_model: str | None = None,
    target_tokens: int = 768,
    overlap_tokens: int = 128,
    log=print,
) -> dict:
    t0 = time.time()
    cdir = Path(root) / slug
    cdir.mkdir(parents=True, exist_ok=True)
    conn = _connect(cdir / "context.sqlite")
    meta_path = cdir / "meta.json"
    old_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    embedder = get_embedder(embedder_spec)
    emb_spec = embedder.spec if embedder else "none"
    chunk_cfg = {"target_tokens": target_tokens, "overlap_tokens": overlap_tokens}
    full_rebuild = old_meta.get("embedder") != emb_spec or old_meta.get("chunking") != chunk_cfg

    old = dict(conn.execute("SELECT doc_id, sha FROM documents").fetchall())
    new_ids = {d.id for d in docs}
    changed = [d for d in docs if full_rebuild or old.get(d.id) != d.sha]
    removed = [i for i in old if i not in new_ids]
    stale = [d.id for d in changed] + removed
    log(f"[curate] {len(docs)} docs: {len(changed)} new/changed, {len(removed)} removed, "
        f"{len(docs) - len(changed)} unchanged")

    for doc_id in stale:
        conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        for t in PER_DOC_TABLES:
            col = "doc_id"
            conn.execute(f"DELETE FROM {t} WHERE {col}=?", (doc_id,))

    entity_names = {d.entity: d.entity_name for d in docs}
    for d in changed:
        conn.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?)",
                     (d.id, d.title, d.category, d.entity, d.entity_name, d.sha, d.content))
        for c in chunk_document(d, target_tokens, overlap_tokens):
            conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?)", (c.id, c.doc_id, c.ord, c.heading, c.text))
        extract_doc(conn, d, entity_names)
    conn.commit()
    log("[curate] deterministic extraction done")

    # --- vectors (reuse unchanged) ---
    rows = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    ids = [r[0] for r in rows]
    n_embedded = 0
    if embedder:
        prev: dict[str, np.ndarray] = {}
        if not full_rebuild and (cdir / "vectors.npy").exists():
            pv = np.load(cdir / "vectors.npy")
            for cid, v in zip(json.loads((cdir / "chunk_ids.json").read_text()), pv):
                prev[cid] = v
        stale_set = set(stale)
        need = [(i, t) for i, t in rows if i not in prev or i.split("#")[0] in stale_set]
        if need:
            log(f"[curate] embedding {len(need)} chunks with {emb_spec} ...")
            vecs = embedder.embed([t for _, t in need])
            for (cid, _), v in zip(need, vecs):
                prev[cid] = v
            n_embedded = len(need)
        mat = np.stack([prev[i] for i in ids]) if ids else np.zeros((0, 1), np.float32)
        np.save(cdir / "vectors.npy", mat)
    (cdir / "chunk_ids.json").write_text(json.dumps(ids))

    build_corpus_artifacts(conn, docs)
    log("[curate] corpus artifacts (fact sheets, rosters, tool cards) built")

    n_llm = 0
    if llm_model and manifest:
        log(f"[curate] LLM manifest extraction with {llm_model} on {len(changed)} docs ...")
        n_llm = llm_extract(conn, docs, manifest, llm_model, only_doc_ids={d.id for d in changed})

    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["documents", "chunks", "entities", "facts", "procedures", "tools", "enums", "edges",
                        "artifacts"]}
    meta = {
        "slug": slug,
        "embedder": emb_spec,
        "chunking": chunk_cfg,
        "manifest": manifest or {},
        "llm_model": llm_model,
        "last_curated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": counts,
        "llm_rows": n_llm,
        "embedded_chunks": n_embedded,
        "build_seconds": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    conn.close()
    log(f"[curate] done in {meta['build_seconds']}s: {counts}")
    return meta


class Context:
    """A loaded, queryable compiled context."""

    def __init__(self, root: str | Path, slug: str, embedder: Embedder | None | str = "auto"):
        self.dir = Path(root) / slug
        if not (self.dir / "context.sqlite").exists():
            raise FileNotFoundError(f"context {slug!r} not compiled under {root}")
        self.slug = slug
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.conn = sqlite3.connect(self.dir / "context.sqlite", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if embedder == "auto":
            embedder = get_embedder(self.meta.get("embedder", "none"))
        self.embedder = embedder
        rows = self.conn.execute("SELECT chunk_id, doc_id, text FROM chunks ORDER BY chunk_id").fetchall()
        ids = [r["chunk_id"] for r in rows]
        vectors = None
        if embedder and (self.dir / "vectors.npy").exists():
            vectors = np.load(self.dir / "vectors.npy")
            if len(vectors) != len(ids):
                vectors = None
        self.chunk_text = {r["chunk_id"]: r["text"] for r in rows}
        self.chunk_doc = {r["chunk_id"]: r["doc_id"] for r in rows}
        self.index = HybridIndex(ids, [r["text"] for r in rows], vectors, embedder)
        self.doc_title = dict(self.conn.execute("SELECT doc_id, title FROM documents").fetchall())

    def sql(self, query: str, params: tuple = ()) -> list[dict]:
        """Read-only SQL over the artifact store."""
        q = query.strip().lower()
        if not (q.startswith("select") or q.startswith("with")):
            raise ValueError("only SELECT queries are allowed")
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def artifact(self, name: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM artifacts WHERE name=?", (name,)).fetchone()
        return dict(r) if r else None

    def read_source(self, doc_id: str) -> str | None:
        r = self.conn.execute("SELECT title, content FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return f"# {r['title']} [{doc_id}]\n\n{r['content']}" if r else None
