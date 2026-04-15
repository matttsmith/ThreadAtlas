"""Static analysis: no network clients in the runtime package.

The spec mandates fully local operation. To make that an enforced property
rather than an aspirational one, we walk the package source and forbid any
import of stdlib networking modules or known third-party HTTP/network
clients. This catches accidental ``urllib.request`` usage etc.

If you genuinely need to add networking later, you must (a) move it to an
opt-in module that is NOT imported by default, and (b) update this allowlist.
"""

from __future__ import annotations

import ast
from pathlib import Path

import threadatlas


# Stdlib networking + popular client libraries we never want to ship with.
FORBIDDEN_TOP_LEVELS = {
    "urllib", "urllib2", "http", "httplib", "ftplib", "smtplib", "poplib",
    "imaplib", "telnetlib", "socket", "socketserver", "ssl", "select",
    "asyncio",  # could be used for sockets; not needed in v1
    # Third parties we never want to import:
    "requests", "httpx", "aiohttp", "urllib3", "websocket", "websockets",
    "boto3", "botocore", "google", "openai", "anthropic", "tiktoken",
}


def _iter_python_files() -> list[Path]:
    pkg_root = Path(threadatlas.__file__).parent
    return list(pkg_root.rglob("*.py"))


def _find_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def test_no_forbidden_imports_in_runtime_package():
    offenders: list[tuple[str, str]] = []
    for py in _iter_python_files():
        rel = py.relative_to(Path(threadatlas.__file__).parent.parent)
        # The MCP module is allowed to depend on the optional 'mcp' package
        # (which is itself a stdio adapter) - but only if a user installs it.
        # Our shipped server uses no third-party imports.
        for top in _find_imports(py):
            if top in FORBIDDEN_TOP_LEVELS:
                offenders.append((str(rel), top))
    assert not offenders, (
        "Forbidden network-related imports found in shipped package:\n"
        + "\n".join(f"  {p}: {m}" for p, m in offenders)
    )


def test_serve_does_not_open_sockets(monkeypatch):
    """Defensive: simulate a socket() call and confirm serve() never reaches it.

    We monkey-patch ``socket.socket`` to raise; if any code path opens a
    socket during ``serve()`` (or its imports), the test fails.
    """
    import socket
    import io
    import threadatlas.mcp.server as mcp_server  # noqa: F401  (already imported)

    sentinel = []

    def _raise(*a, **kw):
        sentinel.append((a, kw))
        raise AssertionError("MCP server attempted to open a socket")

    monkeypatch.setattr(socket, "socket", _raise)
    # Also block higher-level helpers that some libraries use.
    monkeypatch.setattr(socket, "create_connection", _raise, raising=False)

    # We need a vault to call serve(); construct one.
    from threadatlas.core.vault import init_vault
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        v = init_vault(Path(td) / "vault")
        stdin = io.StringIO("")  # EOF immediately
        stdout = io.StringIO()
        rc = mcp_server.serve(v.root, stdin=stdin, stdout=stdout)
        assert rc == 0
    assert sentinel == [], "Socket constructed during MCP serve()"
