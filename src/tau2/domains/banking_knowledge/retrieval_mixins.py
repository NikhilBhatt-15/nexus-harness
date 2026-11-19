"""Retrieval capability MixIns for the banking_knowledge domain.

Each MixIn provides a single retrieval tool via @is_tool methods.
The ToolKitType metaclass collects these tools through MRO when
mixed into a concrete toolkit class.

MixIns define no __init__ — they expect the concrete class to set
the required attributes (e.g., _kb_pipeline, _grep_pipeline, _sandbox)
during its own __init__.
"""

from tau2.environment.toolkit import ToolKitType, ToolType, is_tool


def _format_kb_search_result(pipeline, retrieval_result) -> str:
    """Format timed KB search results for all KB search tool variants."""
    results = retrieval_result.results
    timing = retrieval_result.timing

    if not results:
        output = "No relevant documents found."
        output += f"\n\n[Timing: retrieval={timing.retrieval_ms:.0f}ms"
        if timing.postprocessing_ms > 0:
            output += f", reranking={timing.postprocessing_ms:.0f}ms"
        output += f", total={timing.total_ms:.0f}ms]"
        return output

    formatted = []
    for i, (doc_id, score) in enumerate(results, 1):
        title = pipeline.get_document_title(doc_id) or "Untitled"
        content = pipeline.get_document_content(doc_id) or ""
        formatted.append(
            f"{i}. {title}\n"
            f"   ID: {doc_id}\n"
            f"   Score: {score:.4f}\n"
            f"   Content: {content}\n"
        )

    output = "\n".join(formatted)
    output += f"\n\n[Timing: retrieval={timing.retrieval_ms:.0f}ms"
    if timing.postprocessing_ms > 0:
        output += f", reranking={timing.postprocessing_ms:.0f}ms"
    output += f", total={timing.total_ms:.0f}ms]"
    return output


def _run_kb_search(pipeline, query: str, top_k: int | None = None) -> str:
    """Run a KB search pipeline with timing and shared formatting."""
    retrieve_kwargs = {"return_timing": True}
    if top_k is not None:
        retrieve_kwargs["top_k"] = top_k
    retrieval_result = pipeline.retrieve(query, **retrieve_kwargs)
    return _format_kb_search_result(pipeline, retrieval_result)


class KBSearchMixin(metaclass=ToolKitType):
    """MixIn that provides the KB_search tool.

    Expects ``self._kb_pipeline`` (a RetrievalPipeline) to be set by the
    concrete class before any tool calls.
    """

    @is_tool(ToolType.READ)
    def KB_search(self, query: str) -> str:
        """Search the knowledge base for relevant documents.

        Args:
            query: The search query to find relevant documents

        Returns:
            Relevant document excerpts matching the query
        """
        # TODO: clean up knowledge retrieval pipelines to return structure results
        return _run_kb_search(self._kb_pipeline, query)


class GrepMixin(metaclass=ToolKitType):
    """MixIn that provides the grep tool.

    Expects ``self._grep_pipeline`` (a RetrievalPipeline) to be set.
    """

    @is_tool(ToolType.READ)
    def grep(self, pattern: str) -> str:
        """Search for a regex pattern in all knowledge base documents.

        Returns documents ranked by number of matches, with full content.

        Args:
            pattern: The regex pattern to search for (e.g., 'credit.*card', 'fee|charge')

        Returns:
            Matching documents ranked by relevance (match count)
        """
        results = self._grep_pipeline.retrieve(pattern)

        if not results:
            return f"No matches found for pattern: {pattern}"

        formatted = []
        for i, (doc_id, score) in enumerate(results, 1):
            title = self._grep_pipeline.get_document_title(doc_id) or "Untitled"
            content = self._grep_pipeline.get_document_content(doc_id) or ""
            formatted.append(
                f"{i}. {title}\n"
                f"   ID: {doc_id}\n"
                f"   Score: {score:.4f}\n"
                f"   Content: {content}\n"
            )

        return "\n".join(formatted)


class KBSearchBm25AllToolsMixin(metaclass=ToolKitType):
    """BM25 search for AllTools; expects ``self._kb_bm25_pipeline``."""

    @is_tool(ToolType.READ)
    def KB_search_bm25(self, query: str, k: int = 10) -> str:
        """Search the knowledge base using BM25 sparse retrieval.

        Args:
            query: The search query to find relevant documents.
            k: Maximum number of documents to return (default 10).

        Returns:
            Relevant document excerpts matching the query.
        """
        return _run_kb_search(self._kb_bm25_pipeline, query, top_k=k)


class KBSearchDenseAllToolsMixin(metaclass=ToolKitType):
    """Dense embedding search for AllTools; expects ``self._kb_dense_pipeline``."""

    @is_tool(ToolType.READ)
    def KB_search_dense(self, query: str, k: int = 10) -> str:
        """Search the knowledge base using dense embedding retrieval.

        Args:
            query: The search query to find relevant documents.
            k: Maximum number of documents to return (default 10).

        Returns:
            Relevant document excerpts matching the query.
        """
        return _run_kb_search(self._kb_dense_pipeline, query, top_k=k)


class ShellMixin(metaclass=ToolKitType):
    """MixIn that provides the shell tool.

    Expects ``self._sandbox`` (a SandboxManager) to be set.
    """

    @is_tool(ToolType.READ)
    def shell(self, command: str) -> str:
        """Execute a shell command in the knowledge base directory.

        Use standard Unix utilities to explore and search the knowledge base files.
        Common commands: ls, cat, grep, head, tail, find, wc, awk, sed, etc.

        Args:
            command: The shell command to execute (e.g., "ls -la", "grep -r 'credit card' .", "cat INDEX.txt")

        Returns:
            The command output (stdout) or error message
        """
        if self._sandbox is None:
            return "Error: Sandbox not initialized"

        ret_code, stdout, stderr = self._sandbox.run_command(command)

        if ret_code != 0:
            # grep returns 1 when no matches (not an error)
            if ret_code == 1 and "grep" in command and not stderr:
                return "No matches found."
            if stderr:
                return f"Error (exit code {ret_code}): {stderr}"
            return f"Command failed with exit code {ret_code}"

        return stdout if stdout else "(no output)"


class NexusQueryMixin(metaclass=ToolKitType):
    """MixIn that provides KB_query, backed by a compiled knowledge context.

    Expects ``self._nexus_client`` (a NexusClient) set by the concrete class.
    """

    @is_tool(ToolType.READ)
    def KB_query(self, question: str) -> str:
        """Ask the bank's knowledge service about products, policies, fees, eligibility, or procedures.

        Returns compiled, cited knowledge (product fact sheets, procedure steps with the exact
        internal tool names, and the most relevant policy passages). Ask one specific,
        self-contained question per call and name the exact product, e.g. "What is the monthly
        maintenance fee for the Sky Blue business checking account and how is it waived?".
        For procedures, ask for the ordered steps and the exact tool names and arguments.

        Args:
            question: A specific natural-language question about bank products, policies, fees,
                limits, eligibility, or procedures.

        Returns:
            Compiled knowledge with document citations.
        """
        return self._nexus_client.query(question, task_id=getattr(self, "_task_id", None))


class NexusRouterMixin(metaclass=ToolKitType):
    """Direct access to a compiled nexus-lite context (router mode).

    Expects ``self._nexus_ctx`` (a ``nexus_lite.context.Context``).
    """

    @is_tool(ToolType.READ)
    def KB_orient(self) -> str:
        """Overview of the compiled knowledge: tables, columns, entities and artifact names. Call once first.

        Returns:
            Schema and catalog of the compiled knowledge base.
        """
        ctx = self._nexus_ctx
        tables = ctx.sql("SELECT name, sql FROM sqlite_master WHERE type='table'")
        ents = ctx.sql("SELECT entity, entity_name, category, doc_count FROM entities ORDER BY category, entity")
        lines = ["# Compiled knowledge", "", "## SQL tables (read-only, SQLite)"]
        lines += [f"- {t['sql']}" for t in tables]
        lines += ["", "## Entities (category: entity -> name)"]
        lines += [f"- {e['category']}: {e['entity']} -> {e['entity_name']} ({e['doc_count']} docs)" for e in ents]
        lines += ["", "## Artifacts: `fact_sheet:<entity>`, `roster:<category>`, `tool:<tool_name>` "
                  "(read with KB_read_artifact)",
                  "Tips: facts(kind,text) holds fee/rate/limit lines; procedures holds numbered steps with "
                  "tool_name; tools lists internal tool names with argument hints."]
        return "\n".join(lines)

    @is_tool(ToolType.READ)
    def KB_sql(self, query: str) -> str:
        """Run a read-only SQL SELECT over the compiled knowledge tables (see KB_orient). Max 50 rows.

        Args:
            query: A SQLite SELECT statement.

        Returns:
            Rows as JSON lines, or an error message.
        """
        import json as _json

        try:
            rows = self._nexus_ctx.sql(query)
        except Exception as e:
            return f"SQL error: {e}"
        out = "\n".join(_json.dumps(r, ensure_ascii=False) for r in rows[:50])
        return (out or "(no rows)") + (f"\n... {len(rows) - 50} more rows" if len(rows) > 50 else "")

    @is_tool(ToolType.READ)
    def KB_read_artifact(self, name: str) -> str:
        """Read one compiled artifact, e.g. `fact_sheet:sky_blue`, `roster:credit_cards`, `tool:open_bank_account_4821`.

        Args:
            name: Artifact name.

        Returns:
            The artifact body.
        """
        art = self._nexus_ctx.artifact(name)
        return art["body"] if art else f"No artifact named {name!r}. Use KB_orient to list names."

    @is_tool(ToolType.READ)
    def KB_read_source(self, doc_id: str) -> str:
        """Read the full text of one source document by its id (as cited in other results).

        Args:
            doc_id: Document id, e.g. `doc_credit_cards_ecocard_001`.

        Returns:
            The document text.
        """
        return self._nexus_ctx.read_source(doc_id) or f"No document {doc_id!r}."

    @is_tool(ToolType.READ)
    def KB_search_passages(self, query: str, k: int = 6) -> str:
        """Hybrid (BM25 + dense) search over source passages. Use when SQL/artifacts cannot locate the doc.

        Args:
            query: Natural-language search query.
            k: Number of passages (default 6).

        Returns:
            Ranked passages with their document ids.
        """
        ctx = self._nexus_ctx
        hits = ctx.index.search(query, k=int(k))
        return "\n\n".join(f"{i}. [{ctx.chunk_doc[c]}]\n{ctx.chunk_text[c]}" for i, (c, _) in enumerate(hits, 1)) \
            or "No results."


class RewriteContextMixin(metaclass=ToolKitType):
    """MixIn that provides the rewrite_context tool for summarization."""

    @is_tool(ToolType.WRITE, mutates_state=False)
    def rewrite_context(self, new_context: str) -> str:
        """Replace your working context with a new summary or condensed version.

        Use this after searching to summarize findings, extract key points,
        or condense information for easier reference later in the conversation.

        Args:
            new_context: The summarized or rewritten context to store

        Returns:
            The context you provided, for reference
        """
        return f"Context updated:\n\n{new_context}"
