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
    s.add_argument("--auto-approve", action="store_true",
                   help="Set non-rule-matching imports to 'indexed' "
                        "instead of 'pending_review'. auto_rules.json "
                        "still routes matches to private/quarantined.")
    s.set_defaults(handler=cmd.cmd_import)

    s = sub.add_parser(
        "rescan-rules",
        help="Re-apply auto_rules.json to the existing corpus "
             "(down-classify only; never re-exposes anything).",
    )
    s.add_argument("vault", type=Path)
    s.set_defaults(handler=cmd.cmd_rescan_rules)

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

    s = sub.add_parser(
        "audit-conversation",
        help="Full audit dump of one conversation (messages, chunks, provenance, contributed objects)",
    )
    s.add_argument("vault", type=Path)
    s.add_argument("conversation_id")
    s.set_defaults(handler=cmd.cmd_audit_conversation)

    s = sub.add_parser(
        "audit-object",
        help="Show a derived object and every provenance link (why is this here?)",
    )
    s.add_argument("vault", type=Path)
    s.add_argument("object_id")
    s.set_defaults(handler=cmd.cmd_audit_object)

    s = sub.add_parser(
        "plan-delete",
        help="Preview what hard delete would remove (read-only dry run)",
    )
    s.add_argument("vault", type=Path)
    s.add_argument("conversation_ids", nargs="+")
    s.set_defaults(handler=cmd.cmd_plan_delete)

    s = sub.add_parser(
        "process-approved",
        help="Run chunk + extract across all indexed/private conversations (no state changes)",
    )
    s.add_argument("vault", type=Path)
    s.set_defaults(handler=cmd.cmd_process_approved)

    # --- grouping ---
    s = sub.add_parser(
        "group",
        help="Compute thematic groups (deterministic TF-IDF + k-means). Optional LLM naming with --llm-names.",
    )
    s.add_argument("vault", type=Path)
    s.add_argument("--broad", type=int, default=10)
    s.add_argument("--fine", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--llm-names", action="store_true",
                   help="Use the configured local LLM to name groups (requires local_llm.json with 'group_naming' in used_for)")
    s.set_defaults(handler=cmd.cmd_group)

    s = sub.add_parser("list-groups", help="List thematic groups")
    s.add_argument("vault", type=Path)
    s.add_argument("--level", choices=["broad", "fine"], default=None)
    s.set_defaults(handler=cmd.cmd_list_groups)

    s = sub.add_parser("group-view", help="Show members of one group")
    s.add_argument("vault", type=Path)
    s.add_argument("group_id")
    s.set_defaults(handler=cmd.cmd_group_view)

    # --- LLM features ---
    s = sub.add_parser(
        "summarize",
        help="Generate topical summaries via the configured local LLM (resumable)",
    )
    s.add_argument("vault", type=Path)
    s.add_argument("--conversation-id", default=None,
                   help="If set, summarize only this conversation")
    s.add_argument("--limit", type=int, default=None,
                   help="Max conversations to summarize in this run")
    s.add_argument("--force", action="store_true",
                   help="Re-summarize even conversations that already have an LLM summary")
    s.set_defaults(handler=cmd.cmd_summarize)

    s = sub.add_parser(
        "llm-chunk",
        help="Refine chunk boundaries with the configured local LLM (merges only; never adds splits)",
    )
    s.add_argument("vault", type=Path)
    s.add_argument("--conversation-id", default=None)
    s.set_defaults(handler=cmd.cmd_llm_chunk)

    # --- operator hygiene ---
    s = sub.add_parser("check", help="Vault health check: normalized files, FTS sync, leaks, orphans")
    s.add_argument("vault", type=Path)
    s.set_defaults(handler=cmd.cmd_check)

    s = sub.add_parser(
        "rebuild-from-normalized",
        help="Disaster recovery: rebuild DB from vault/normalized/ JSON files",
    )
    s.add_argument("vault", type=Path)
    s.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    s.set_defaults(handler=cmd.cmd_rebuild_from_normalized)

    s = sub.add_parser("report", help="Generate a static HTML report (no server; just a file)")
    s.add_argument("vault", type=Path)
    s.add_argument("--out", type=Path, default=None,
                   help="Output path (default: vault/reports/report_<ts>.html)")
    s.set_defaults(handler=cmd.cmd_report)

    s = sub.add_parser("tag", help="Add one or more manual tags to a conversation")
    s.add_argument("vault", type=Path)
    s.add_argument("conversation_id")
    s.add_argument("tags", nargs="+")
    s.set_defaults(handler=cmd.cmd_tag)

    s = sub.add_parser("untag", help="Remove one or more manual tags from a conversation")
    s.add_argument("vault", type=Path)
    s.add_argument("conversation_id")
    s.add_argument("tags", nargs="+")
    s.set_defaults(handler=cmd.cmd_untag)

    # --- canonicalization ---
    s = sub.add_parser("obj-merge",
                       help="Merge one or more derived objects into a winner (same kind only)")
    s.add_argument("vault", type=Path)
    s.add_argument("winner")
    s.add_argument("losers", nargs="+")
    s.set_defaults(handler=cmd.cmd_obj_merge)

    s = sub.add_parser("obj-rename", help="Rename a derived object")
    s.add_argument("vault", type=Path)
    s.add_argument("object_id")
    s.add_argument("title")
    s.set_defaults(handler=cmd.cmd_obj_rename)

    s = sub.add_parser("obj-suppress",
                       help="Mark a derived object suppressed (hidden from listings)")
    s.add_argument("vault", type=Path)
    s.add_argument("object_id")
    s.add_argument("--unsuppress", action="store_true",
                   help="Revert a previous suppression")
    s.set_defaults(handler=cmd.cmd_obj_suppress)

    # --- manual project linking ---
    s = sub.add_parser("link", help="Set a conversation's primary_project_id")
    s.add_argument("vault", type=Path)
    s.add_argument("conversation_id")
    s.add_argument("project_id")
    s.set_defaults(handler=cmd.cmd_link)

    s = sub.add_parser("unlink", help="Clear a conversation's primary_project_id")
    s.add_argument("vault", type=Path)
    s.add_argument("conversation_id")
    s.set_defaults(handler=cmd.cmd_unlink)

    # --- TUI ---
    s = sub.add_parser("tui", help="Interactive ASCII dashboard (read-only curses UI)")
    s.add_argument("vault", type=Path)
    s.add_argument("--preview", default=None,
                   metavar="SCREEN",
                   choices=["overview", "conversations", "groups", "projects",
                            "open_loops", "decisions", "entities", "help"],
                   help="Print a single screen to stdout (no curses) and exit")
    s.set_defaults(handler=cmd.cmd_tui)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
