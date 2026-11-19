"""curate(): compile sources into a Context.

Two extractor families, both writing into the Context's SQLite store:

1. Deterministic (free, always on). Rules over markdown structure:
   - ``entities``    one row per grouped entity (product / program)
   - ``facts``       fee / rate / limit / bonus / eligibility lines with numbers
   - ``procedures``  numbered steps under a heading, with tool names pulled out
   - ``tools``       registry of internal tool names, docs and argument hints
   - ``enums``       quoted literal values and SNAKE/UPPER codes with context
   - ``edges``       entity -> entity mentions across documents
   - ``artifacts``   markdown artifacts: per-entity fact sheets, category rosters,
                     tool cards, procedure cards

2. LLM manifest extraction (optional, ``--llm-model``). Same idea as a Nexus
   manifest: each ``artifact_types[]`` entry with ``format: sqlite`` becomes a
   table ``x_<name>`` whose columns the model fills per document.

Compilation is incremental: documents are keyed by content hash; unchanged
documents keep their chunks, vectors and extracted rows.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict

from nexus_lite.corpus import Document

TOOL_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d{4})\b")
QUOTED_RE = re.compile(r"['\"‘“]([A-Z][^'\"’”\n]{1,60})['\"’”]")
CODE_RE = re.compile(r"`([^`\n]{2,80})`")
UPPER_RE = re.compile(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)\b")
SNAKE_RE = re.compile(r"\b([a-z]+(?:_[a-z]+){1,6})\b")
NUM_RE = re.compile(r"\$\s?[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?\s?%|\b\d[\d,]*(?:\.\d+)?\b")
STEP_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")

FACT_KINDS = [
    ("apy", r"\bapy\b|annual percentage yield|interest rate|\bapr\b"),
    ("annual_fee", r"annual fee"),
    ("monthly_fee", r"monthly (maintenance |service )?fee|maintenance fee"),
    ("atm", r"\batm\b"),
    ("wire", r"\bwire"),
    ("overdraft", r"overdraft|nsf"),
    ("foreign", r"foreign"),
    ("cashback", r"cash ?back|rewards? rate|points? per|\bpoints\b|\bmiles\b"),
    ("bonus", r"bonus|welcome offer|promo|promotion|sign-?up"),
    ("limit", r"\blimit|maximum|minimum|\bup to\b|at least|no more than"),
    ("deposit", r"deposit"),
    ("eligibility", r"eligib|qualif|must be|require"),
    ("fee", r"\bfee"),
    ("time", r"\bdays?\b|business days|months?\b|years?\b"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(doc_id TEXT PRIMARY KEY, title TEXT, category TEXT, entity TEXT,
    entity_name TEXT, sha TEXT, content TEXT);
CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT, ord INT, heading TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS entities(entity TEXT PRIMARY KEY, entity_name TEXT, category TEXT, doc_count INT,
    aliases TEXT, titles TEXT);
CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, entity TEXT, category TEXT, doc_id TEXT, heading TEXT,
    kind TEXT, text TEXT, numbers TEXT);
CREATE TABLE IF NOT EXISTS procedures(id INTEGER PRIMARY KEY, doc_id TEXT, entity TEXT, procedure TEXT,
    step_ord INT, text TEXT, tool_name TEXT);
CREATE TABLE IF NOT EXISTS tools(tool_name TEXT PRIMARY KEY, doc_ids TEXT, mention_count INT, arg_hints TEXT,
    contexts TEXT);
CREATE TABLE IF NOT EXISTS enums(id INTEGER PRIMARY KEY, value TEXT, kind TEXT, doc_id TEXT, context TEXT);
CREATE TABLE IF NOT EXISTS edges(src TEXT, dst TEXT, doc_id TEXT, kind TEXT);
CREATE TABLE IF NOT EXISTS artifacts(name TEXT PRIMARY KEY, type TEXT, scope TEXT, entity TEXT, category TEXT,
    body TEXT, sources TEXT);
CREATE INDEX IF NOT EXISTS ix_facts_entity ON facts(entity);
CREATE INDEX IF NOT EXISTS ix_proc_doc ON procedures(doc_id);
CREATE INDEX IF NOT EXISTS ix_enums_value ON enums(value);
"""

PER_DOC_TABLES = ["facts", "procedures", "enums", "edges"]


def _lines_with_heading(content: str):
    heading = ""
    for line in content.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            heading = m.group(1).strip()
            continue
        if line.strip():
            yield heading, line.strip()


def _clean(line: str) -> str:
    return re.sub(r"^[-*+]\s+", "", line).strip()


def classify_fact(text: str) -> str | None:
    low = text.lower()
    if not NUM_RE.search(text) and "free" not in low and "waiv" not in low and "no " not in low:
        return None
    for kind, pat in FACT_KINDS:
        if re.search(pat, low):
            return kind
    return "number" if re.search(r"\$|%", text) else None


def extract_doc(conn: sqlite3.Connection, doc: Document, entity_names: dict[str, str]) -> None:
    """Deterministic per-document extraction."""
    cur = conn.cursor()
    procedure_ord: dict[str, int] = defaultdict(int)
    for heading, line in _lines_with_heading(doc.content):
        text = _clean(line)
        kind = classify_fact(text)
        if kind:
            cur.execute(
                "INSERT INTO facts(entity,category,doc_id,heading,kind,text,numbers) VALUES(?,?,?,?,?,?,?)",
                (doc.entity, doc.category, doc.id, heading, kind, text, json.dumps(NUM_RE.findall(text))),
            )
        m = STEP_RE.match(line)
        if m:
            proc = heading or doc.title
            procedure_ord[proc] += 1
            tools = TOOL_RE.findall(text)
            cur.execute(
                "INSERT INTO procedures(doc_id,entity,procedure,step_ord,text,tool_name) VALUES(?,?,?,?,?,?)",
                (doc.id, doc.entity, f"{doc.title} :: {proc}", procedure_ord[proc], m.group(2).strip(),
                 ",".join(dict.fromkeys(tools))),
            )
        for value in QUOTED_RE.findall(line):
            cur.execute("INSERT INTO enums(value,kind,doc_id,context) VALUES(?,?,?,?)",
                        (value.strip(), "quoted", doc.id, text[:300]))
        for value in UPPER_RE.findall(line):
            cur.execute("INSERT INTO enums(value,kind,doc_id,context) VALUES(?,?,?,?)",
                        (value, "code", doc.id, text[:300]))
        for value in CODE_RE.findall(line):
            cur.execute("INSERT INTO enums(value,kind,doc_id,context) VALUES(?,?,?,?)",
                        (value, "code_span", doc.id, text[:300]))
        for value in SNAKE_RE.findall(line):
            if not TOOL_RE.fullmatch(value) and value.count("_") >= 1 and len(value) > 6:
                cur.execute("INSERT INTO enums(value,kind,doc_id,context) VALUES(?,?,?,?)",
                            (value, "snake", doc.id, text[:300]))
    low = doc.content.lower()
    for ent, name in entity_names.items():
        if ent != doc.entity and len(name) > 3 and name.lower() in low:
            cur.execute("INSERT INTO edges(src,dst,doc_id,kind) VALUES(?,?,?,?)",
                        (doc.entity, ent, doc.id, "mentions"))


def build_corpus_artifacts(conn: sqlite3.Connection, docs: list[Document], max_sheet_chars: int = 24000) -> None:
    """Rebuild corpus-scope tables/artifacts (cheap; always recomputed)."""
    cur = conn.cursor()
    for t in ("entities", "tools", "artifacts"):
        cur.execute(f"DELETE FROM {t}")

    by_entity: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        by_entity[d.entity].append(d)

    # --- entities + aliases (title prefixes before ':' that recur) ---
    for ent, ds in by_entity.items():
        prefixes = Counter(d.title.split(":")[0].strip() for d in ds if ":" in d.title)
        aliases = [p for p, c in prefixes.most_common(5) if len(p) < 60]
        cur.execute(
            "INSERT INTO entities VALUES(?,?,?,?,?,?)",
            (ent, ds[0].entity_name, ds[0].category, len(ds), json.dumps(aliases),
             json.dumps([d.title for d in ds])),
        )

    # --- tools registry ---
    tool_docs: dict[str, set] = defaultdict(set)
    tool_ctx: dict[str, list] = defaultdict(list)
    tool_args: dict[str, Counter] = defaultdict(Counter)
    arg_decl = re.compile(r"(?:\*\*|`)([a-z][a-z0-9_]*)(?:\*\*|`)\s*\(|^\s*(?:\d+[.)]|[-*])\s+([a-z][a-z0-9_]*)\s*\(")
    for d in docs:
        doc_tools = set(TOOL_RE.findall(d.content))
        if doc_tools:  # argument declarations like "**transaction_id** (string)" or "6. phone (string)"
            for line in d.content.splitlines():
                m = arg_decl.search(line)
                arg = m and (m.group(1) or m.group(2))
                if arg and arg not in doc_tools:
                    for t in doc_tools:
                        tool_args[t][arg] += 1
        for heading, line in _lines_with_heading(d.content):
            for t in TOOL_RE.findall(line):
                tool_docs[t].add(d.id)
                if len(tool_ctx[t]) < 12:
                    tool_ctx[t].append(f"[{d.id}] {d.title} > {heading}: {_clean(line)}")
                for a in re.findall(r"\b([a-z]+(?:_[a-z]+)+)\s*(?:=|:)", line):
                    if a != t:
                        tool_args[t][a] += 1
                for a in CODE_RE.findall(line):
                    if a != t and re.fullmatch(r"[a-z_]+", a):
                        tool_args[t][a] += 1
    for t, ds in tool_docs.items():
        cur.execute(
            "INSERT INTO tools VALUES(?,?,?,?,?)",
            (t, json.dumps(sorted(ds)), len(tool_ctx[t]), json.dumps([a for a, _ in tool_args[t].most_common(20)]),
             json.dumps(tool_ctx[t])),
        )

    # --- per-entity fact sheets (the "company-level fact sheet" pattern) ---
    def doc_num(d: Document) -> int:
        m = re.search(r"_(\d+)$", d.id)
        return int(m.group(1)) if m else 0

    for ent, ds in by_entity.items():
        ds = sorted(ds, key=doc_num)
        name = ds[0].entity_name
        head = [f"# Fact sheet: {name} ({ds[0].category.replace('_', ' ')})", ""]
        key = cur.execute(
            "SELECT kind, text, doc_id FROM facts WHERE entity=? AND kind NOT IN ('time','eligibility','limit') "
            "ORDER BY id LIMIT 40", (ent,)).fetchall()
        if key:
            head.append("## Key figures")
            head += [f"- ({k}) {t} [{doc}]" for k, t, doc in key]
            head.append("")
        tools = sorted({t for d in ds for t in TOOL_RE.findall(d.content)})
        if tools:
            head.append("## Internal tools referenced: " + ", ".join(tools))
            head.append("")
        body = "\n".join(head) + "\n## Source documents\n"
        for d in ds:
            section = f"\n### {d.title} [{d.id}]\n{d.content}\n"
            if len(body) + len(section) > max_sheet_chars:
                body += f"\n(... {len(ds)} docs total; remaining omitted for budget)\n"
                break
            body += section
        cur.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)",
                    (f"fact_sheet:{ent}", "fact_sheet", "entity", ent, ds[0].category, body,
                     json.dumps([d.id for d in ds])))

    # --- category rosters ("All X") ---
    by_cat: dict[str, list[str]] = defaultdict(list)
    for ent, ds in by_entity.items():
        by_cat[ds[0].category].append(ent)
    for cat, ents in by_cat.items():
        lines = [f"# Roster: all {cat.replace('_', ' ')}", ""]
        sources = []
        for ent in sorted(ents):
            name = by_entity[ent][0].entity_name
            lines.append(f"## {name}  (entity `{ent}`, {len(by_entity[ent])} docs)")
            rows = cur.execute(
                "SELECT kind, text, doc_id FROM facts WHERE entity=? AND kind IN "
                "('apy','annual_fee','monthly_fee','cashback','bonus','foreign','atm','overdraft','fee') "
                "ORDER BY id LIMIT 14", (ent,)).fetchall()
            for k, t, doc in rows:
                lines.append(f"- ({k}) {t} [{doc}]")
                sources.append(doc)
            lines.append("")
        cur.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)",
                    (f"roster:{cat}", "roster", "category", None, cat, "\n".join(lines),
                     json.dumps(sorted(set(sources)))))

    # --- tool cards ---
    for t, ds in tool_docs.items():
        steps = cur.execute(
            "SELECT procedure, step_ord, text, doc_id FROM procedures WHERE tool_name LIKE ? LIMIT 20",
            (f"%{t}%",)).fetchall()
        lines = [f"# Tool card: {t}", "", f"Documents: {', '.join(sorted(ds))}",
                 f"Argument hints: {', '.join(a for a, _ in tool_args[t].most_common(20)) or '(none found)'}",
                 "", "## Mentions"] + [f"- {c}" for c in tool_ctx[t]]
        if steps:
            lines += ["", "## Procedure steps using this tool"] + [
                f"- {p} step {o}: {s} [{doc}]" for p, o, s, doc in steps]
        cur.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)",
                    (f"tool:{t}", "tool_card", "tool", None, None, "\n".join(lines), json.dumps(sorted(ds))))
    conn.commit()


# ---------------------------------------------------------------------------
# Optional: LLM manifest extraction (Nexus-manifest-shaped)
# ---------------------------------------------------------------------------

def llm_extract(conn: sqlite3.Connection, docs: list[Document], manifest: dict, model: str,
                only_doc_ids: set[str] | None = None) -> int:
    """Fill ``x_<type>`` tables per the manifest using an LLM. Returns rows written."""
    import litellm

    types = [t for t in manifest.get("curate", {}).get("artifacts", {}).get("artifact_types", [])
             if t.get("format") == "sqlite"]
    written = 0
    for t in types:
        cols = [c["name"] for c in t["columns"]]
        if "source_doc" not in cols:
            cols.append("source_doc")
        conn.execute(f"CREATE TABLE IF NOT EXISTS x_{t['name']}({', '.join(c + ' TEXT' for c in cols)})")
        col_desc = "\n".join(f"- {c['name']}: {c.get('description', '')}" for c in t["columns"])
        for d in docs:
            if only_doc_ids is not None and d.id not in only_doc_ids:
                continue
            prompt = (
                f"Extract `{t['name']}` rows from the document. {t.get('description', '')}\n"
                f"Columns:\n{col_desc}\n\nReturn JSON: {{\"rows\": [{{column: value, ...}}]}}. "
                "Use exact strings from the document; leave a column empty if not stated. "
                "Return {\"rows\": []} if nothing applies.\n\n"
                f"DOCUMENT {d.id}\n# {d.title}\n{d.content[:50000]}"
            )
            resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}],
                                      response_format={"type": "json_object"}, temperature=0)
            try:
                rows = json.loads(resp.choices[0].message.content).get("rows", [])
            except (json.JSONDecodeError, AttributeError):
                rows = []
            conn.execute(f"DELETE FROM x_{t['name']} WHERE source_doc=?", (d.id,))
            for r in rows:
                r["source_doc"] = d.id
                conn.execute(f"INSERT INTO x_{t['name']}({', '.join(cols)}) VALUES({', '.join('?' * len(cols))})",
                             [str(r.get(c, "") if r.get(c) is not None else "") for c in cols])
                written += 1
        conn.commit()
    return written
