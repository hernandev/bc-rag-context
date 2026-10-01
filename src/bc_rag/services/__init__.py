"""Background services: qdrant, indexer, and mcp, kept up by one supervisor.

Commands that need the index call `ensure_stack` through this package attribute, so
tests can replace it with `monkeypatch.setattr("bc_rag.services.ensure_stack", ...)`.
"""

from __future__ import annotations

from bc_rag.services.state import SERVICE_NAMES
from bc_rag.services.supervisor import StackStatus, ensure_stack

__all__ = ["SERVICE_NAMES", "StackStatus", "ensure_stack"]
