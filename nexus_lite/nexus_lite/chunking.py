"""Heading-aware chunking.

Token counts are approximated as ``len(text) / 4`` (no tokenizer dependency).
Each chunk carries its document title and nearest heading path so a chunk is
self-describing when retrieved alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nexus_lite.corpus import Document

CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    id: str
    doc_id: str
    ord: int
    heading: str
    text: str  # includes the "title > heading" prefix line


def _sections(content: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, body) sections."""
    out: list[tuple[str, str]] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    current = ""
    for line in content.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if buf and "".join(buf).strip():
                out.append((current, "\n".join(buf).strip()))
            buf = []
            level = len(m.group(1))
            path = [(lvl, h) for lvl, h in path if lvl < level] + [(level, m.group(2).strip())]
            current = " > ".join(h for _, h in path)
            continue
        buf.append(line)
    if buf and "".join(buf).strip():
        out.append((current, "\n".join(buf).strip()))
    return out


def chunk_document(doc: Document, target_tokens: int = 768, overlap_tokens: int = 128) -> list[Chunk]:
    target = target_tokens * CHARS_PER_TOKEN
    overlap = overlap_tokens * CHARS_PER_TOKEN
    pieces: list[tuple[str, str]] = []  # (heading, text)
    cur_head, cur = "", ""
    for heading, body in _sections(doc.content) or [("", doc.content)]:
        block = (f"## {heading}\n" if heading else "") + body
        if cur and len(cur) + len(block) + 2 > target:
            pieces.append((cur_head, cur))
            tail = cur[-overlap:] if overlap else ""
            cur, cur_head = (tail + "\n\n" + block).strip(), heading
        else:
            cur = (cur + "\n\n" + block).strip() if cur else block
            cur_head = cur_head or heading
        while len(cur) > target * 1.5:  # very long single section: hard split
            pieces.append((cur_head, cur[:target]))
            cur = cur[target - overlap :]
    if cur:
        pieces.append((cur_head, cur))
    chunks = []
    for i, (heading, text) in enumerate(pieces):
        prefix = doc.title + (f" > {heading}" if heading else "")
        chunks.append(
            Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, ord=i, heading=heading, text=f"[{prefix}]\n{text}")
        )
    return chunks
