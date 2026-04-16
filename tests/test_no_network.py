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


# The llama_server_backend is the ONE module allowed to use stdlib HTTP
# (urllib) for loopback-only communication with a locally-running model
# server.  It is never imported at module load time — the runner does a
# lazy ``from .llama_server_backend import ...`` only when the provider is
# ``llama_server``.  Every other module must remain network-free.
_NETWORK_ALLOWLIST = {
    "threadatlas/llm/llama_server_backend.py",
}


def test_no_forbidden_imports_in_runtime_package():
    offenders: list[tuple[str, str]] = []
    for py in _iter_python_files():
        rel = py.relative_to(Path(threadatlas.__file__).parent.parent)
        rel_str = str(rel).replace("\\", "/")  # normalize Windows paths
        if rel_str in _NETWORK_ALLOWLIST:
            continue
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


def test_llm_runner_does_not_open_sockets(monkeypatch, tmp_vault):
    """The LLM runner must only use subprocess - never network.

    Patches ``socket.socket`` to raise; if any code path opens a socket
    while the LLM is invoked, the test fails.
    """
    import json as _json
    import socket as _socket
    import sys as _sys

    sentinel: list = []

    def _block(*a, **kw):
        sentinel.append((a, kw))
        raise AssertionError("LLM runner attempted to open a socket")

    monkeypatch.setattr(_socket, "socket", _block)
    monkeypatch.setattr(_socket, "create_connection", _block, raising=False)

    (tmp_vault.root / "local_llm.json").write_text(_json.dumps({
        "command": [_sys.executable, "-m", "tests.fake_llm", "summary_ok"],
        "timeout_seconds": 10,
        "max_prompt_chars": 2000,
        "max_response_chars": 1000,
        "used_for": ["summaries"],
    }), encoding="utf-8")

    from threadatlas.llm import LLMRunner, load_config
    cfg = load_config(tmp_vault.root)
    runner = LLMRunner(tmp_vault, cfg)
    resp = runner.run("summaries", "ping")
    assert resp.success
    assert sentinel == []


def test_llama_server_backend_is_not_imported_by_default():
    """The llama_server_backend module must NOT be imported at package load time.

    It uses urllib (networking). The runner only imports it lazily when the
    provider is 'llama_server'. Verify that importing threadatlas.llm does
    NOT pull in llama_server_backend.
    """
    import importlib
    import sys

    # Unload if previously cached.
    mod_name = "threadatlas.llm.llama_server_backend"
    was_loaded = mod_name in sys.modules
    if was_loaded:
        saved = sys.modules.pop(mod_name)

    try:
        # Re-import the package fresh.
        importlib.reload(importlib.import_module("threadatlas.llm"))
        assert mod_name not in sys.modules, (
            "llama_server_backend is imported at module load time — "
            "it should only be imported lazily when provider=llama_server"
        )
    finally:
        if was_loaded:
            sys.modules[mod_name] = saved


def test_subprocess_runner_does_not_import_llama_server(monkeypatch, tmp_vault):
    """When using the subprocess provider, llama_server_backend must never load."""
    import json as _json
    import sys as _sys

    (tmp_vault.root / "local_llm.json").write_text(_json.dumps({
        "command": [_sys.executable, "-m", "tests.fake_llm", "summary_ok"],
        "timeout_seconds": 10,
        "max_prompt_chars": 2000,
        "max_response_chars": 1000,
        "used_for": ["summaries"],
    }), encoding="utf-8")

    mod_name = "threadatlas.llm.llama_server_backend"
    was_loaded = mod_name in _sys.modules
    if was_loaded:
        saved = _sys.modules.pop(mod_name)

    try:
        from threadatlas.llm import LLMRunner, load_config
        cfg = load_config(tmp_vault.root)
        runner = LLMRunner(tmp_vault, cfg)
        resp = runner.run("summaries", "ping")
        assert resp.success
        assert mod_name not in _sys.modules, (
            "llama_server_backend was imported during a subprocess run"
        )
    finally:
        if was_loaded:
            _sys.modules[mod_name] = saved


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
