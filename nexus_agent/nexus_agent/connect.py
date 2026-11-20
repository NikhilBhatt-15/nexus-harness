"""`nxa connect`: turn a dataset or database into a ready-to-use agent project.

    nxa connect ./my_csvs --name sales                       # folder of CSV/TSV/XLSX/Parquet/JSON files
    nxa connect data.db --name shop                          # existing SQLite file (used in place, read-only)
    nxa connect postgresql://user@host/db --name crm         # any SQLAlchemy URL (pip install sqlalchemy + driver)
    nxa connect ./csvs --name hc --dictionary columns.csv    # merge an existing data dictionary
    nxa connect ./csvs --name hc --describe-model gemini/gemini-3.1-flash-lite   # LLM-written column notes

Pipeline (each step prints what it did):
  1. load       files -> SQLite (typed, chunked); databases are attached as-is
  2. profile    per table: rows; per column: type, missing %, distinct, range, top values, placeholders
  3. relations  join keys discovered from names + uniqueness + value overlap (one-to-many / one-to-one)
  4. docs       one markdown doc per table + an overview with the join map and data quirks;
                descriptions merged from a dictionary file and/or written by an LLM
  5. knowledge  docs compiled into a nexus-lite knowledge base
  6. project    agent.yaml, persistent Python plugin (read-only DB connection), analysis skill
  7. check      self-test: can the knowledge base answer "how do X and Y join" / "what is column C"?
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

FILE_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".ndjson"}
SAMPLE_ROWS = 200_000
ID_NAME = re.compile(r"(^id$|_id$|^id_|^sk_id|_key$|^key$|_code$|^uuid|_uuid$|_no$|_number$)", re.I)


def slugify(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return ("t_" + s) if s[:1].isdigit() else (s or "table")


def singular(word: str) -> str:
    w = word.lower()
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes"):
        return w[:-2]
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


# ---------------------------------------------------------------------------
# database access (sqlite3 or SQLAlchemy behind one tiny interface)
# ---------------------------------------------------------------------------


class DB:
    def __init__(self, url: str, read_only: bool = True):
        self.url = url
        if "://" in url and not url.startswith("sqlite:///"):
            from sqlalchemy import create_engine, inspect  # optional dependency

            self.kind = "sqlalchemy"
            self.engine = create_engine(url)
            self._inspect = inspect(self.engine)
        else:
            path = Path(url.removeprefix("sqlite:///")).resolve()
            self.kind = "sqlite"
            self.path = path
            uri = f"file:{path.as_posix()}?mode=ro" if read_only else path.as_posix()
            self.conn = sqlite3.connect(uri, uri=read_only, check_same_thread=False)

    def rows(self, sql: str) -> list[tuple]:
        if self.kind == "sqlite":
            return self.conn.execute(sql).fetchall()
        from sqlalchemy import text

        with self.engine.connect() as c:
            return [tuple(r) for r in c.execute(text(sql))]

    def tables(self) -> list[str]:
        if self.kind == "sqlite":
            return [r[0] for r in self.rows("SELECT name FROM sqlite_master WHERE type='table' "
                                            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return sorted(self._inspect.get_table_names())

    def columns(self, table: str) -> list[tuple[str, str]]:
        if self.kind == "sqlite":
            return [(r[1], r[2] or "") for r in self.rows(f'PRAGMA table_info("{table}")')]
        return [(c["name"], str(c["type"])) for c in self._inspect.get_columns(table)]

    def q(self, name: str) -> str:
        return f'"{name}"'

    def sample_from(self, table: str, n_rows: int) -> str:
        """A FROM clause over at most SAMPLE_ROWS rows, spread across the table where the DB allows it."""
        if self.kind == "sqlite" and n_rows > SAMPLE_ROWS:
            k = math.ceil(n_rows / SAMPLE_ROWS)
            return f'(SELECT * FROM "{table}" WHERE rowid % {k} = 0)'
        return f'(SELECT * FROM "{table}" LIMIT {SAMPLE_ROWS})' if n_rows > SAMPLE_ROWS else f'"{table}"'


# ---------------------------------------------------------------------------
# 1. load
# ---------------------------------------------------------------------------


def _readers(path: Path) -> Callable[[], Any]:
    import pandas as pd

    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        return lambda: pd.read_csv(path, chunksize=250_000, low_memory=False, encoding_errors="replace")
    if ext == ".tsv":
        return lambda: pd.read_csv(path, sep="\t", chunksize=250_000, low_memory=False, encoding_errors="replace")
    if ext in (".jsonl", ".ndjson"):
        return lambda: pd.read_json(path, lines=True, chunksize=250_000)
    if ext == ".json":
        return lambda: [pd.read_json(path)]
    if ext in (".xlsx", ".xls"):
        def read_excel():
            try:
                sheets = pd.read_excel(path, sheet_name=None)
            except ImportError as e:
                raise SystemExit(f"Reading {path.name} needs openpyxl: pip install openpyxl ({e})")
            return [df for df in sheets.values()]
        return read_excel
    if ext == ".parquet":
        def read_parquet():
            try:
                return [pd.read_parquet(path)]
            except ImportError as e:
                raise SystemExit(f"Reading {path.name} needs pyarrow: pip install pyarrow ({e})")
        return read_parquet
    raise ValueError(path)


def load_files(source: Path, db_path: Path, log: Callable[[str], None], skip: set[str] | None = None) -> list[str]:
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in FILE_EXTS)
    files = [f for f in files if f.name not in (skip or set())]
    if not files:
        raise SystemExit(f"No data files ({', '.join(sorted(FILE_EXTS))}) found in {source}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    loaded = []
    for f in files:
        table = slugify(f.stem)
        t0 = time.time()
        con.execute(f'DROP TABLE IF EXISTS "{table}"')
        n = 0
        for chunk in _readers(f)():
            chunk.columns = [str(c).strip() or f"col_{i}" for i, c in enumerate(chunk.columns)]
            chunk.to_sql(table, con, if_exists="append", index=False)
            n += len(chunk)
        con.commit()
        loaded.append(table)
        log(f"  loaded {f.name} -> {table}: {n:,} rows ({time.time() - t0:.0f}s)")
    con.close()
    return loaded


# ---------------------------------------------------------------------------
# 2. profile
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    name: str
    type: str
    missing_pct: float = 0.0
    distinct: int = 0
    unique_in_sample: bool = False
    min: Any = None
    max: Any = None
    mean: float | None = None
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    placeholder: Any = None
    description: str = ""
    described_by: str = ""
    is_key: bool = False


@dataclass
class TableProfile:
    name: str
    rows: int
    columns: list[ColumnProfile]
    note: str = ""


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def profile_db(db: DB, tables: list[str], log: Callable[[str], None]) -> list[TableProfile]:
    out = []
    for t in tables:
        t0 = time.time()
        n = db.rows(f'SELECT COUNT(*) FROM "{t}"')[0][0]
        src = db.sample_from(t, n)
        if db.kind == "sqlite" and n > SAMPLE_ROWS:
            try:  # materialize the sample once instead of rescanning the big table for every column
                db.conn.execute("DROP TABLE IF EXISTS temp._nxa_sample")
                db.conn.execute(f"CREATE TEMP TABLE _nxa_sample AS SELECT * FROM {src}")
                src = "temp._nxa_sample"
            except sqlite3.OperationalError:
                pass
        sample_n = min(n, SAMPLE_ROWS) if n else 0
        cols = []
        for name, ctype in db.columns(t):
            c = ColumnProfile(name=name, type=ctype)
            qn = db.q(name)
            nulls, distinct, mn, mx = db.rows(
                f"SELECT SUM(CASE WHEN {qn} IS NULL OR {qn} = '' THEN 1 ELSE 0 END), COUNT(DISTINCT {qn}), "
                f"MIN({qn}), MAX({qn}) FROM {src}")[0]
            real_n = db.rows(f"SELECT COUNT(*) FROM {src}")[0][0] if n > SAMPLE_ROWS else n
            c.missing_pct = round(100 * (nulls or 0) / real_n, 1) if real_n else 0.0
            c.distinct = distinct or 0
            c.unique_in_sample = bool(real_n) and c.distinct == real_n - (nulls or 0) and c.distinct > 1
            c.min, c.max = mn, mx
            if _num(mn) is not None and _num(mx) is not None and not isinstance(mn, str):
                c.mean = _num(db.rows(f"SELECT AVG({qn}) FROM {src}")[0][0])
            if 0 < c.distinct <= 30 or (isinstance(mn, str) and c.distinct <= 2000):
                c.top_values = [(r[0], r[1]) for r in db.rows(
                    f"SELECT {qn}, COUNT(*) AS k FROM {src} WHERE {qn} IS NOT NULL GROUP BY {qn} ORDER BY k DESC LIMIT 12")]
            elif c.mean is not None and real_n:
                # a single extreme value repeated many times looks like a placeholder for "missing"
                v, k = db.rows(f"SELECT {qn}, COUNT(*) AS k FROM {src} WHERE {qn} IS NOT NULL GROUP BY {qn} "
                               f"ORDER BY k DESC LIMIT 1")[0]
                nv = _num(v)
                if nv is not None and k / real_n > 0.02 and nv in (_num(mn), _num(mx)) and c.distinct > 30:
                    # compare the gap to the spread of all OTHER values (the placeholder inflates the mean)
                    lo, hi = db.rows(f"SELECT MIN({qn}), MAX({qn}) FROM {src} WHERE {qn} IS NOT NULL AND {qn} <> {v}")[0]
                    lo, hi = _num(lo), _num(hi)
                    if lo is not None and hi is not None:
                        gap = (nv - hi) if nv == _num(mx) else (lo - nv)
                        if gap > 5 * max(1.0, hi - lo):
                            c.placeholder = v
            c.is_key = bool(ID_NAME.search(name)) and not (c.mean is not None and c.distinct <= 30)
            cols.append(c)
        out.append(TableProfile(name=t, rows=n, columns=cols,
                                note=f"profiled on {sample_n:,} sampled rows" if n > SAMPLE_ROWS else ""))
        log(f"  profiled {t}: {n:,} rows, {len(cols)} columns ({time.time() - t0:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# 3. relationships
# ---------------------------------------------------------------------------


@dataclass
class Relation:
    parent: str
    parent_col: str
    child: str
    child_col: str
    kind: str  # one-to-many | one-to-one | many-to-many
    overlap: float
    child_rows_per_parent: float


def _is_unique(db: DB, t: str, col: str) -> bool:
    n, d = db.rows(f'SELECT COUNT({db.q(col)}), COUNT(DISTINCT {db.q(col)}) FROM "{t}"')[0]
    return bool(n) and n == d


def discover_relations(db: DB, profiles: list[TableProfile], log: Callable[[str], None]) -> list[Relation]:
    by_table = {p.name: p for p in profiles}
    cols = {p.name: {c.name.lower(): c for c in p.columns} for p in profiles}
    candidates: list[tuple[str, str, str, str]] = []  # (t1, c1, t2, c2) same values expected
    names = list(by_table)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for cname, ca in cols[a].items():
                cb = cols[b].get(cname)
                if cb and (ca.is_key or cb.is_key or ca.unique_in_sample or cb.unique_in_sample):
                    candidates.append((a, ca.name, b, cb.name))
    # differently named foreign keys: orders.customer_id -> customers.id
    for t in names:
        if "id" in cols[t]:
            for other in names:
                if other == t:
                    continue
                fk = cols[other].get(f"{singular(t)}_id")
                if fk:
                    candidates.append((t, cols[t]["id"].name, other, fk.name))
    rels: list[Relation] = []
    seen = set()
    unique_cache: dict[tuple[str, str], bool] = {}

    def uniq(t: str, c: str) -> bool:
        if (t, c) not in unique_cache:
            unique_cache[(t, c)] = _is_unique(db, t, c)
        return unique_cache[(t, c)]

    for a, ca, b, cb in candidates:
        key = tuple(sorted([(a, ca), (b, cb)]))
        if key in seen:
            continue
        seen.add(key)
        ua, ub = uniq(a, ca), uniq(b, cb)
        if not ua and not ub:
            # two children of the same parent (both reference a key that is unique somewhere else): not a relation
            if any(t not in (a, b) and ca.lower() in cols[t] and uniq(t, cols[t][ca.lower()].name) for t in names):
                continue
        if ua and not ub:
            parent, pc, child, cc = a, ca, b, cb
        elif ub and not ua:
            parent, pc, child, cc = b, cb, a, ca
        elif ua and ub:
            stem = re.sub(r"(_id|id|_key|_code|_no|_number)$", "", ca.lower()).rstrip("_")
            if stem and singular(b) == singular(stem) and singular(a) != singular(stem):
                parent, pc, child, cc = b, cb, a, ca  # orders.order_id is the parent of order_items.order_id
            elif stem and singular(a) == singular(stem):
                parent, pc, child, cc = a, ca, b, cb
            else:
                parent, pc, child, cc = (a, ca, b, cb) if by_table[a].rows <= by_table[b].rows else (b, cb, a, ca)
        else:
            parent, pc, child, cc = (a, ca, b, cb) if by_table[a].rows <= by_table[b].rows else (b, cb, a, ca)
        vals = [r[0] for r in db.rows(f'SELECT DISTINCT {db.q(cc)} FROM "{child}" WHERE {db.q(cc)} IS NOT NULL LIMIT 2000')]
        if not vals:
            continue
        lits = ",".join(repr(v) if isinstance(v, str) else str(v) for v in vals)
        found = db.rows(f'SELECT COUNT(DISTINCT {db.q(pc)}) FROM "{parent}" WHERE {db.q(pc)} IN ({lits})')[0][0]
        overlap = found / len(vals)
        if overlap < 0.3:
            continue
        crow = by_table[child].rows
        pdist = db.rows(f'SELECT COUNT(DISTINCT {db.q(cc)}) FROM "{child}"')[0][0] or 1
        per = crow / pdist
        kind = "many-to-many" if not (ua or ub) else ("one-to-one" if per <= 1.01 else "one-to-many")
        rels.append(Relation(parent, pc, child, cc, kind, round(overlap, 3), round(per, 2)))
        by_table[parent].columns[[c.name for c in by_table[parent].columns].index(pc)].is_key = True
        by_table[child].columns[[c.name for c in by_table[child].columns].index(cc)].is_key = True
        log(f"  relation {parent}.{pc} -> {child}.{cc}: {kind}, {overlap:.0%} of keys match, "
            f"{per:.1f} {child} rows per {pc}")
    return rels


def ensure_indexes(db_path: Path, rels: list[Relation], log: Callable[[str], None]) -> None:
    con = sqlite3.connect(db_path)
    made = 0
    for r in rels:
        for t, c in ((r.parent, r.parent_col), (r.child, r.child_col)):
            con.execute(f'CREATE INDEX IF NOT EXISTS "ix_{t}_{c}" ON "{t}"("{c}")')
            made += 1
    con.commit()
    con.close()
    log(f"  indexed {made} join columns")


# ---------------------------------------------------------------------------
# 4. dictionary + docs
# ---------------------------------------------------------------------------


def _expand_table_names(raw: str) -> list[str]:
    raw = re.sub(r"\.(csv|tsv|txt|parquet|xlsx|json)$", "", raw.strip(), flags=re.I)
    m = re.search(r"\{([^}]*)\}", raw)
    if m:
        return [slugify(raw[: m.start()] + opt + raw[m.end():]) for opt in m.group(1).split("|")]
    return [slugify(raw)]


def load_dictionary(path: Path, profiles: list[TableProfile]) -> int:
    """Merge a data dictionary (CSV/TSV with column + description, optional table) into the profiles."""
    import pandas as pd

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        d = pd.read_csv(path, sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        d = pd.read_csv(path, sep=sep, encoding="latin-1")
    lower = {c.lower().strip(): c for c in d.columns}
    col_c = next((lower[k] for k in ("column", "field", "row", "name", "column_name", "variable") if k in lower), None)
    desc_c = next((lower[k] for k in ("description", "desc", "meaning", "definition", "comment") if k in lower), None)
    tab_c = next((lower[k] for k in ("table", "table_name", "file", "dataset") if k in lower), None)
    note_c = next((lower[k] for k in ("special", "notes", "note") if k in lower), None)
    if not col_c or not desc_c:
        raise SystemExit(f"{path.name}: need a column-name column and a description column (got {list(d.columns)})")
    by_name = {p.name: p for p in profiles}
    merged = 0
    for _, r in d.iterrows():
        tables = _expand_table_names(str(r[tab_c])) if tab_c else list(by_name)
        desc = str(r[desc_c]).strip()
        if note_c and isinstance(r[note_c], str) and r[note_c].strip():
            desc += f" ({r[note_c].strip()})"
        for t in tables:
            p = by_name.get(t) or next((by_name[n] for n in by_name if n.startswith(t) or t.startswith(n)), None)
            if not p:
                continue
            for c in p.columns:
                if c.name.lower() == str(r[col_c]).strip().lower() and not c.description:
                    c.description, c.described_by = desc, "dictionary"
                    merged += 1
    return merged


def describe_with_llm(profiles: list[TableProfile], model: str, log: Callable[[str], None]) -> int:
    from nexus_agent.llm import LLM

    llm = LLM(model, temperature=0)
    n = 0
    for p in profiles:
        todo = [c for c in p.columns if not c.description]
        if not todo:
            continue
        lines = [f"- {c.name} ({c.type}): " + ", ".join(
            x for x in [f"examples {[v for v, _ in c.top_values[:5]]}" if c.top_values else "",
                        f"range {c.min}..{c.max}" if c.min is not None else "",
                        f"{c.missing_pct}% missing"] if x) for c in todo[:150]]
        prompt = (f"Table `{p.name}` ({p.rows:,} rows) has these columns. For each, write a short plain-English "
                  "description of what it most likely means (say 'unclear' if you cannot tell). Return JSON "
                  '{"column_name": "description", ...}.\n' + "\n".join(lines))
        try:
            text = llm.complete([{"role": "user", "content": prompt}]).text
            data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        except Exception as e:
            log(f"  LLM descriptions for {p.name} failed: {e}")
            continue
        for c in todo:
            if isinstance(data.get(c.name), str) and data[c.name].strip():
                c.description, c.described_by = data[c.name].strip(), "llm (unverified)"
                n += 1
        log(f"  described {p.name} with {model}")
    return n


def _fmt(v: Any) -> str:
    f = _num(v)
    if f is not None and not isinstance(v, str):
        return f"{f:,.0f}" if abs(f) >= 1000 and f == int(f) else f"{f:,.4g}"
    return str(v)[:60]


def write_docs(docs_dir: Path, name: str, profiles: list[TableProfile], rels: list[Relation]) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for old in docs_dir.glob("*.md"):
        old.unlink()
    written = []
    for p in profiles:
        out_rel = [r for r in rels if r.parent == p.name]
        in_rel = [r for r in rels if r.child == p.name]
        keys = [c.name for c in p.columns if c.is_key]
        lines = [f"# Table {p.name}", "", f"SQL table name: `{p.name}`. {p.rows:,} rows, {len(p.columns)} columns."
                 + (f" ({p.note})" if p.note else ""), ""]
        if keys:
            lines.append(f"Key columns: {', '.join(f'`{k}`' for k in keys)}.")
        if in_rel or out_rel:
            lines += ["", "## How it joins"]
            for r in in_rel:
                agg = (f" Aggregate {p.name} per {r.child_col} before joining to {r.parent} to avoid double counting."
                       if r.kind != "one-to-one" else "")
                lines.append(f"- `{p.name}.{r.child_col}` -> `{r.parent}.{r.parent_col}` ({r.kind}: about "
                             f"{r.child_rows_per_parent} {p.name} rows per {r.parent} row).{agg}")
            for r in out_rel:
                lines.append(f"- `{p.name}.{r.parent_col}` <- `{r.child}.{r.child_col}` ({r.kind}).")
        lines += ["", "## Columns", ""]
        for c in p.columns:
            facts = [c.type or "?"]
            if c.missing_pct:
                facts.append(f"{c.missing_pct}% missing")
            facts.append(f"{c.distinct:,} distinct")
            if c.top_values and c.distinct <= 30 and not c.is_key:
                facts.append("values: " + ", ".join(f"{_fmt(v)} ({k:,})" for v, k in c.top_values[:10]))
            elif c.min is not None and not (c.is_key and isinstance(c.min, str)):
                facts.append(f"range {_fmt(c.min)} to {_fmt(c.max)}" + (f", mean {_fmt(c.mean)}" if c.mean is not None else ""))
            if c.placeholder is not None:
                facts.append(f"NOTE: value {_fmt(c.placeholder)} looks like a placeholder for missing data")
            desc = f" {c.description}" + (f" [{c.described_by}]" if c.described_by.startswith("llm") else "") \
                if c.description else ""
            lines.append(f"- **{c.name}**:{desc} ({'; '.join(facts)})")
        path = docs_dir / f"table_{p.name}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)

    ov = [f"# Dataset guide: {name}", "", "## Tables", ""]
    ov += [f"- `{p.name}`: {p.rows:,} rows, {len(p.columns)} columns" for p in profiles]
    ov += ["", "## How the tables join", ""]
    if rels:
        for r in rels:
            ov.append(f"- `{r.child}.{r.child_col}` -> `{r.parent}.{r.parent_col}`: {r.kind} "
                      f"({r.overlap:.0%} of {r.child} keys found in {r.parent}; {r.child_rows_per_parent} rows per key)")
        ov += ["", "Rule: when a child table has many rows per key, aggregate it per key (COUNT, SUM, AVG, MAX) "
                   "first, then join to the parent; joining raw rows multiplies the parent's rows."]
    else:
        ov.append("No join keys were detected between tables.")
    quirks = []
    for p in profiles:
        for c in p.columns:
            if c.placeholder is not None:
                quirks.append(f"- `{p.name}.{c.name}`: value {_fmt(c.placeholder)} is probably a placeholder for missing.")
            if c.missing_pct >= 50:
                quirks.append(f"- `{p.name}.{c.name}` is {c.missing_pct}% missing.")
            if c.distinct == 1 and p.rows > 1:
                quirks.append(f"- `{p.name}.{c.name}` has a single value ({_fmt(c.min)}); it carries no information.")
    if quirks:
        ov += ["", "## Data quirks", ""] + quirks[:80] + ([f"- ... and {len(quirks) - 80} more"] if len(quirks) > 80 else [])
    path = docs_dir / "guide_overview.md"
    path.write_text("\n".join(ov) + "\n", encoding="utf-8")
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# 6. project files
# ---------------------------------------------------------------------------

PLUGIN = '''"""Persistent Python analysis session for {name} (generated by nxa connect)."""

from pathlib import Path

from nexus_agent.toolsets import python_session

HERE = Path(__file__).resolve().parent.parent
DB_URL = r"{db_url}"

PRELUDE = f"""
import sqlite3, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
DB_URL = r"{{DB_URL}}"
{connect_code}

def q(sql, params=()):
    \\"\\"\\"Run a read-only SQL query and return a DataFrame.\\"\\"\\"
    return pd.read_sql_query(sql, conn, params=params)
"""

toolset = python_session(
    HERE / "workspace",
    prelude=PRELUDE,
    name="analysis",
    auto_approve_readonly=True,
    description=("Already defined: pd, np, `conn` (read-only connection to the {name} database), `q(sql)` -> "
                 "DataFrame, DB_URL. Never open another connection or hard-code paths."),
)
'''

SKILL = """---
name: {name}-analysis
description: How to analyze the {name} data correctly (joins, quirks, reporting)
---
1. Use kb_query to check what tables and columns mean, how tables join, and known data quirks before writing SQL.
2. Join tables only on the documented keys. When the child side has many rows per key, aggregate it per key first.
3. Respect the quirks listed in the dataset guide (placeholder values, heavily missing columns).
4. Prefer SQL aggregation (db_query / q()) over loading whole tables; big tables: filter or aggregate first.
5. Report numbers with counts; small groups are noisy.
6. Put a short table of results in the final answer and say which tables/columns you used.
"""


def write_project(out: Path, name: str, db_url: str, kb_root: Path, read_only_note: str) -> None:
    (out / "tools").mkdir(parents=True, exist_ok=True)
    (out / "workspace").mkdir(exist_ok=True)
    skill_dir = out / "skills" / f"{name}-analysis"
    skill_dir.mkdir(parents=True, exist_ok=True)
    if "://" in db_url and not db_url.startswith("sqlite:///"):
        connect_code = ("from sqlalchemy import create_engine\n"
                        "conn = create_engine(DB_URL, execution_options=dict(postgresql_readonly=True)).connect()")
        sql_entry = f"  - sql: {{url: '{db_url}', name: db, read_only: true}}"
    else:
        path = db_url.removeprefix("sqlite:///")
        connect_code = "conn = sqlite3.connect('file:' + DB_URL + '?mode=ro', uri=True, check_same_thread=False)  # read-only"
        rel = Path(path).resolve()
        try:
            rel_s = rel.relative_to(out.resolve()).as_posix()
        except ValueError:
            rel_s = rel.as_posix()
        sql_entry = f"  - sqlite: {{path: {rel_s}, name: db, read_only: true}}"
        db_url = rel.as_posix()
    (out / "tools" / "analysis.py").write_text(
        PLUGIN.format(name=name, db_url=db_url, connect_code=connect_code), encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(SKILL.format(name=name), encoding="utf-8")
    try:
        kb_rel = Path(kb_root).resolve().relative_to(out.resolve()).as_posix()
    except ValueError:
        kb_rel = Path(kb_root).resolve().as_posix()
    yaml_text = f"""# Generated by `nxa connect`. Web UI: nxa serve -c {out.name}/agent.yaml   Terminal: nxa chat -c {out.name}/agent.yaml
# Paths are relative to this file. {read_only_note}
model: gemini/gemini-3.1-flash-lite
permissions: auto          # reads + read-only Python run freely; writes ask
max_steps: 40
workdir: workspace
skills_dirs: [skills]
system: |
  You are a data analyst for the {name} dataset. Answer with numbers from the data, not guesses.
  Use kb_query for what tables/columns mean, how tables join and data quirks; db_schema / db_query for SQL;
  run_python (persistent session) for pandas analysis or modeling. Report counts with rates.
toolsets:
{sql_entry}
  - knowledge: {{slug: {name}, source: docs, root: {kb_rel}, target_tokens: 300, overlap_tokens: 40}}
  - python: {{path: tools/analysis.py}}
  - files: {{root: workspace}}
  - memory: {{path: .nxa/memory.json}}
"""
    (out / "agent.yaml").write_text(yaml_text, encoding="utf-8")
    gi = out / ".gitignore"
    if not gi.exists():
        gi.write_text("data/\nworkspace/\n.nxa/\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. self-check
# ---------------------------------------------------------------------------


def self_check(kb_root: Path, slug: str, profiles: list[TableProfile], rels: list[Relation],
               log: Callable[[str], None]) -> dict:
    from nexus_lite.query import Engine

    eng = Engine(str(kb_root))
    checks = []
    for r in rels[:15]:
        q = f"How does the {r.child} table join to the {r.parent} table?"
        checks.append(("join", q, [r.child_col, r.parent]))
    described = [(p.name, c) for p in profiles for c in p.columns if c.description and len(c.description) > 15]
    for t, c in described[:: max(1, len(described) // 15)][:15]:
        checks.append(("column", f"What does the column {c.name} in {t} mean?", [c.name]))
    for p in profiles[:10]:
        checks.append(("table", f"How many rows does the {p.name} table have?", [f"{p.rows:,}"]))
    ok = 0
    for kind, q, must in checks:
        text = eng.query({"scope": [slug], "ask": q, "budget": {"max_chars": 6000}})["output"][0]["content"][0]["text"]
        hit = all(m.lower() in text.lower() for m in must)
        ok += hit
        if not hit:
            log(f"  MISS [{kind}] {q}")
    res = {"checks": len(checks), "passed": ok}
    log(f"  knowledge self-check: {ok}/{len(checks)} answered from the knowledge base")
    return res


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def connect(source: str, name: str, out: str | None = None, dictionary: str | None = None,
            describe_model: str | None = None, kb_root: str | None = None, embedder: str | None = None,
            log: Callable[[str], None] = print) -> dict:
    t_start = time.time()
    name = slugify(name)
    out_dir = Path(out or name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = Path(kb_root).resolve() if kb_root else out_dir / ".nxa" / "contexts"
    src_path = Path(source)
    is_url = "://" in source and not source.startswith("sqlite:///")
    created_db = False

    log("1/7 load")
    if is_url:
        db_url, note = source, "Database attached in place (read-only)."
        log(f"  attached {source.split('@')[-1]}")
    elif src_path.suffix.lower() in (".db", ".sqlite", ".sqlite3") or source.startswith("sqlite:///"):
        db_url, note = str(Path(source.removeprefix("sqlite:///")).resolve()), "SQLite file used in place (read-only)."
        log(f"  using {db_url}")
    else:
        db_path = out_dir / "data" / f"{name}.db"
        skip = {Path(dictionary).name} if dictionary else set()
        load_files(src_path, db_path, log, skip=skip)
        db_url, note = str(db_path), f"Data loaded from {src_path.resolve()} into data/{name}.db."
        created_db = True

    db = DB(db_url, read_only=True)
    tables = db.tables()
    log(f"2/7 profile ({len(tables)} tables)")
    profiles = profile_db(db, tables, log)

    log("3/7 relations")
    rels = discover_relations(db, profiles, log)
    if db.kind == "sqlite" and rels and created_db:
        try:
            ensure_indexes(Path(db_url), rels, log)
        except sqlite3.OperationalError as e:
            log(f"  could not add indexes ({e}); joins may be slower")

    log("4/7 docs")
    if dictionary:
        log(f"  merged {load_dictionary(Path(dictionary), profiles)} descriptions from {Path(dictionary).name}")
    if describe_model:
        log(f"  {describe_with_llm(profiles, describe_model, log)} columns described by {describe_model}")
    docs = write_docs(out_dir / "docs", name, profiles, rels)
    log(f"  wrote {len(docs)} docs to {out_dir / 'docs'}")

    log("5/7 knowledge base")
    from nexus_lite.context import compile_context
    from nexus_lite.corpus import load_corpus

    meta = compile_context(kb, name, load_corpus(out_dir / "docs"), embedder_spec=embedder, target_tokens=300,
                           overlap_tokens=40, log=lambda m: None)
    log(f"  compiled {meta['counts']['documents']} docs, {meta['counts']['chunks']} chunks ({meta['embedder']})")

    log("6/7 project files")
    write_project(out_dir, name, db_url, kb, note)
    log(f"  agent.yaml, tools/analysis.py, skills/{name}-analysis/SKILL.md")

    log("7/7 self-check")
    check = self_check(kb, name, profiles, rels, log)

    summary = {"name": name, "project": str(out_dir), "db": db_url, "tables": {p.name: p.rows for p in profiles},
               "relations": [r.__dict__ for r in rels], "self_check": check,
               "seconds": round(time.time() - t_start, 1)}
    (out_dir / "profile.json").write_text(json.dumps(
        {"summary": summary, "tables": [{"name": p.name, "rows": p.rows, "columns": [c.__dict__ for c in p.columns]}
                                        for p in profiles]}, indent=1, default=str), encoding="utf-8")
    log(f"\nready in {summary['seconds']}s -> {out_dir}\n  web UI:   nxa serve -c {out_dir / 'agent.yaml'}\n"
        f"  terminal: nxa chat -c {out_dir / 'agent.yaml'}")
    return summary
