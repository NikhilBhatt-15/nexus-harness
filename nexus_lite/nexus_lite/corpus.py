"""Source loading and entity grouping.

A source document is ``{id, title, content}``. JSON files with those keys and
plain ``.md`` files (id = stem, title = first H1 or stem) are both accepted.

Entity grouping maps each document to ``(category, entity)``. The default rule
fits ids like ``doc_<category>_<entity>_<NNN>`` given a list of known
categories (longest prefix wins); anything else falls back to the first token.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Banking (tau-Knowledge) categories. Longest match wins, so
# ``business_checking_accounts`` beats ``checking_accounts``.
DEFAULT_CATEGORIES = [
    "bank_accounts",
    "business_checking_accounts",
    "business_credit_cards",
    "business_savings_accounts",
    "buy_now_pay_later",
    "checking_accounts",
    "credit_cards",
    "customer_support",
    "everyone_pay",
    "personal_subscriptions",
    "savings_accounts",
]

_SMALL_WORDS = {"and", "or", "of", "the", "for", "to", "a", "an", "in", "on"}


@dataclass
class Document:
    id: str
    title: str
    content: str
    category: str = ""
    entity: str = ""  # slug, e.g. "sky_blue"
    entity_name: str = ""  # display, e.g. "Sky Blue"
    meta: dict = field(default_factory=dict)

    @property
    def sha(self) -> str:
        return hashlib.sha256(f"{self.title}\n{self.content}".encode()).hexdigest()[:16]

    def as_markdown(self) -> str:
        return f"# {self.title}\n\n{self.content}\n"


def humanize(slug: str) -> str:
    words = slug.replace("-", " - ").replace("_", " ").split()
    out = []
    for i, w in enumerate(words):
        if w.startswith("(") and w.endswith(")"):
            out.append("(" + w[1:-1].lower() + ")")
        elif i and w in _SMALL_WORDS:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out).replace(" - ", "-")


def group_document(doc_id: str, categories: list[str]) -> tuple[str, str]:
    stem = re.sub(r"^doc_", "", doc_id)
    stem = re.sub(r"_\d+$", "", stem)
    for cat in sorted(categories, key=len, reverse=True):
        if stem == cat:
            return cat, cat
        if stem.startswith(cat + "_"):
            return cat, stem[len(cat) + 1 :]
    head, _, rest = stem.partition("_")
    return head, rest or head


def load_corpus(path: str | Path, categories: list[str] | None = None) -> list[Document]:
    """Load every ``*.json`` / ``*.md`` document under ``path`` (sorted, deterministic)."""
    root = Path(path)
    cats = categories or DEFAULT_CATEGORIES
    docs: list[Document] = []
    for f in sorted(root.rglob("*")):
        if f.suffix == ".json":
            raw = json.loads(f.read_text(encoding="utf-8"))
            doc = Document(id=raw["id"], title=raw.get("title", raw["id"]), content=raw["content"])
        elif f.suffix == ".md":
            text = f.read_text(encoding="utf-8")
            m = re.match(r"#\s+(.+)\n", text)
            title = m.group(1).strip() if m else f.stem
            body = text[m.end() :] if m else text
            doc = Document(id=f.stem, title=title, content=body.strip())
        else:
            continue
        doc.category, doc.entity = group_document(doc.id, cats)
        doc.entity_name = humanize(doc.entity)
        docs.append(doc)
    return docs
