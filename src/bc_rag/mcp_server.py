"""One stdio MCP server for every cataloged project.

Agents register `bc-rag mcp` once. `search` takes an optional project name.
Omit it and the query runs across every living root in `~/.bc-rag/catalog.json`.
"""

from __future__ import annotations

from typing import Any

from bc_rag.catalog import living_projects
from bc_rag.query import search_projects
from bc_rag.store import Hit


def run_mcp() -> None:
    from mcp.server import MCPServer

    mcp = MCPServer(
        "bc-rag",
        instructions=(
            "Search local source and docs indexed by bc-rag-context. "
            "Call list_projects to see cataloged repos, then search. "
            "Pass project to restrict to one repo, or omit it to search all."
        ),
    )

    @mcp.tool()
    def list_projects() -> list[dict[str, str]]:
        """List indexed projects this MCP server can search."""
        return [{"name": e.name, "root": e.root} for e in living_projects()]

    @mcp.tool()
    def search(
        query: str,
        project: str | None = None,
        limit: int = 8,
        groups: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Hybrid semantic + BM25 search over indexed project context.

        Args:
            query: Natural language or identifier search.
            project: Optional catalog name or absolute root. Omit to search all.
            limit: Maximum hits to return.
            groups: Optional config group names. Only hits from these groups.
            tags: Sidecar filters as key:value (vendor:olo). All must match.
        """
        result = search_projects(
            query,
            project=project,
            limit=limit,
            use_rerank=True,
            groups=groups,
            tags=tags,
        )
        return {
            "query": result.query,
            "project": project,
            "groups": groups,
            "tags": tags,
            "hits": [_hit_dict(hit) for hit in result.hits],
        }

    @mcp.tool()
    def list_tags(project: str | None = None) -> dict[str, Any]:
        """List tags and groups stored on indexed points, with point counts.

        Args:
            project: Optional catalog name or absolute root. Defaults to the first catalog entry.
        """
        from pathlib import Path

        from bc_rag.catalog import find_project
        from bc_rag.runtime import open_session

        entries = living_projects()
        if project:
            found = find_project(project, entries)
            if found is None:
                raise ValueError(f"unknown project: {project}")
            entries = [found]
        if not entries:
            return {"tags": [], "groups": []}
        entry = entries[0]
        session = open_session(Path(entry.root), need_reranker=False, write=False)
        try:
            tags = session.store.facet_values("tags")
            groups = session.store.facet_values("group")
        finally:
            session.close()
        return {
            "project": entry.name,
            "tags": [{"tag": name, "points": count} for name, count in tags],
            "groups": [{"group": name, "points": count} for name, count in groups],
        }

    mcp.run(transport="stdio")


def _hit_dict(hit: Hit) -> dict[str, Any]:
    return {
        "project": hit.project,
        "path": hit.path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "language": hit.language,
        "kind": hit.kind,
        "symbol": hit.symbol,
        "heading_path": hit.heading_path,
        "group": (hit.payload or {}).get("group"),
        "tags": (hit.payload or {}).get("tags") or [],
        "score": hit.score,
        "text": hit.text,
    }
