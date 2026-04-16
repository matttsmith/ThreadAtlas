#!/usr/bin/env python3
"""MCP server entry point for the Claude Desktop extension.

This is the script that Claude Desktop launches via ``uv run``.  It
accepts the vault path from the desktop extension's user_config and
delegates to the ThreadAtlas stdio MCP server.

Usage (invoked automatically by Claude Desktop):
    uv run --directory <extension_dir> server/main.py --vault /path/to/vault
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ThreadAtlas MCP server (stdio)",
    )
    parser.add_argument(
        "--vault",
        type=Path,
        required=True,
        help="Absolute path to the ThreadAtlas vault directory",
    )
    args = parser.parse_args()

    vault_path = args.vault.expanduser().resolve()
    if not vault_path.exists():
        print(
            f"Vault directory does not exist: {vault_path}\n"
            f"Create one first with: threadatlas init {vault_path}",
            file=sys.stderr,
        )
        return 1

    from threadatlas.mcp.server import serve
    return serve(vault_path)


if __name__ == "__main__":
    sys.exit(main())
