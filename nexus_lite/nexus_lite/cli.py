"""nexus-lite CLI.

    nexus-lite build  --source DIR --slug banking [--embedder ollama:embeddinggemma-2] [--manifest m.json --llm-model M]
    nexus-lite query  --slug banking "What is the Sky Blue monthly fee?" [--model M] [--depth deep]
    nexus-lite sql    --slug banking "SELECT * FROM tools LIMIT 5"
    nexus-lite serve  [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_ROOT = os.environ.get("NEXUS_LITE_ROOT", "contexts")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nexus-lite")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="directory holding compiled contexts")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="curate(): compile sources into a context (incremental)")
    b.add_argument("--source", required=True)
    b.add_argument("--slug", required=True)
    b.add_argument("--embedder", default=None, help="ollama:<model> | litellm:<model> | none")
    b.add_argument("--manifest", default=None)
    b.add_argument("--llm-model", default=None)
    b.add_argument("--target-tokens", type=int, default=768)
    b.add_argument("--overlap-tokens", type=int, default=128)

    q = sub.add_parser("query", help="run one KnowQL-lite query")
    q.add_argument("--slug", required=True)
    q.add_argument("ask")
    q.add_argument("--model", default=None)
    q.add_argument("--depth", default="standard")
    q.add_argument("--max-chars", type=int, default=None)
    q.add_argument("--json", action="store_true")

    s = sub.add_parser("sql", help="read-only SQL over a context's artifact store")
    s.add_argument("--slug", required=True)
    s.add_argument("query")

    v = sub.add_parser("serve", help="HTTP API (Nexus v0-compatible query surface)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)

    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.cmd == "build":
        from nexus_lite.context import compile_context
        from nexus_lite.corpus import load_corpus

        docs = load_corpus(args.source)
        manifest = json.loads(open(args.manifest, encoding="utf-8").read()) if args.manifest else None
        compile_context(args.root, args.slug, docs, embedder_spec=args.embedder, manifest=manifest,
                        llm_model=args.llm_model, target_tokens=args.target_tokens,
                        overlap_tokens=args.overlap_tokens)
    elif args.cmd == "query":
        from nexus_lite.query import Engine

        req = {"scope": [args.slug], "ask": args.ask, "budget": {"depth": args.depth}}
        if args.max_chars:
            req["budget"]["max_chars"] = args.max_chars
        if args.model:
            req["model"] = args.model
        res = Engine(args.root).query(req)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(res["output"][0]["content"][0]["text"] if res["status"] == "completed" else res)
            print("\nplan:", json.dumps(res.get("plan")), "\nusage:", res.get("usage"),
                  "\ncitations:", [c["id"] for c in res.get("citations", [])])
    elif args.cmd == "sql":
        from nexus_lite.context import Context

        for row in Context(args.root, args.slug, embedder=None).sql(args.query):
            print(json.dumps(row, ensure_ascii=False))
    elif args.cmd == "serve":
        from nexus_lite.server import serve

        serve(args.root, args.host, args.port)


if __name__ == "__main__":
    main()
