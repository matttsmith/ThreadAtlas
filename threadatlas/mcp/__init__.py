"""Local MCP adapter (stdio only).

This module exposes a small read-mostly tool surface to Claude. It refuses to
expose any conversation that is not in ``indexed`` state.

We deliberately implement a minimal MCP-compatible JSON-RPC protocol over
stdio rather than depending on a third-party SDK; the spec requires "no
runtime dependency that silently fetches resources" and the standard library
is enough.
"""

from .server import serve, build_tools

__all__ = ["serve", "build_tools"]
