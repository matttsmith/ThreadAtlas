"""ThreadAtlas command-line interface.

Built on argparse from the standard library to keep the dependency surface
minimal. Subcommand handlers live in :mod:`commands`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import __version__
from . import commands as cmd


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="threadatlas",
        description=f"ThreadAtlas {__version__}: fully local conversation intelligence",
    )
    p.add_argument("--version", action="version", version=f"threadatlas {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="Initialize a new vault directory")
    s.add_argument("vault", type=Path)
    s.set_defaults(handler=cmd.cmd_init)

    s = sub.add_parser("import", help="Import an export (file/dir/zip)")
    s.add_argument("vault", type=Path)
    s.add_argument("path", type=Path, help="Path to export file/dir/zip")
    s.add_argument("--source", default="auto", choices=["auto", "chatgpt", "claude"],
                   help="Source format. Default: auto-detect")
    s.add_argument("--no-copy", action="store_true",
                   help="Do not copy the input into raw_imports/")
    s.set_defaults(handler=cmd.cmd_import)

    s = sub.add_parser("review", help="List conversations awaiting review")
    s.add_argument("vault", type=Path)
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--state", default="pending_review",
                   help="State to list (default: pending_review)")
    s.set_defaults(handler=cmd.cmd_review)

    for action, target_state in [
        ("approve", "indexed"),
        ("private", "private"),
        ("quarantine", "quarantined"),
    ]:
        s = sub.add_parser(action, help=f"Move conversation(s) to {target_state}")
        s.add_argument("vault", type=Path)
        s.add_argument("conversation_ids", nargs="+")
        s.set_defaults(handler=cmd.make_state_handler(target_state))

    s = sub.add_parser("delete", help="Hard-delete one or more conversations (irreversible)")
    s.add_argument("vault", type=Path)
    s.add_argument("conversation_ids", nargs="+")
    s.add_argument("--yes", action="store_true",
                   help="Skip confirmation prompt")
    s.set_defaults(handler=cmd.cmd_delete)

    s = sub.add_parser("chunk", help="(Re)build chunks for eligible conversations")
    s.add_argument("vault", type=Path)
    s.add_argument("--conversation-id", default=None,
                   help="If set, chunk only this conversation")
    s.set_defaults(handler=cmd.cmd_chunk)

    s = sub.add_parser("extract", help="Run heuristic extraction across eligible conversations")
    s.add_argument("vault", type=Path)
    s.add_argument("--conversation-id", default=None)
    s.set_defaults(handler=cmd.cmd_extract)

    s = sub.add_parser("rebuild-index", help="Drop and rebuild all FTS indexes")
    s.add_argument("vault", type=Path)
    s.set_defaults(handler=cmd.cmd_rebuild_index)

    s = sub.add_parser("search", help="Keyword search")
    s.add_argument("vault", type=Path)
    s.add_argument("query")
    s.add_argument("--mode", choices=["conversations", "chunks"], default="conversations")
    s.add_argument("--include-private", action="store_true",
                   help="Search private content as well (CLI-only)")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(handler=cmd.cmd_search)

    s = sub.add_parser("inspect", help="Show what is stored about a conversation")
    s.add_argument("vault", type=Path)
    s.add_argument("conversation_id")
    s.set_defaults(handler=cmd.cmd_inspect)

    s = sub.add_parser("list-projects", help="List active project derived objects")
    s.add_argument("vault", type=Path)
    s.add_argument("--include-private", action="store_true")
    s.set_defaults(handler=cmd.cmd_list_projects)

    s = sub.add_parser("project", help="Show a project page")
    s.add_argument("vault", type=Path)
    s.add_argument("project_id")
    s.add_argument("--include-private", action="store_true")
    s.set_defaults(handler=cmd.cmd_project)

    s = sub.add_parser("export", help="Export a workbook to XLSX")
    s.add_argument("vault", type=Path)
    s.add_argument("--profile", default="review_workbook",
                   choices=cmd.list_export_profiles())
    s.add_argument("--out", type=Path, default=None,
                   help="Output path (default: vault/exports/<profile>_<ts>.xlsx)")
    s.set_defaults(handler=cmd.cmd_export)

    s = sub.add_parser("mcp", help="Run the local stdio MCP server")
    s.add_argument("vault", type=Path)
    s.set_defaults(handler=cmd.cmd_mcp)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
