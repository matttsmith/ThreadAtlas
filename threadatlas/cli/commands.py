"""CLI subcommand handlers."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..audit import audit_conversation, audit_object, plan_hard_delete
from ..cluster import regroup_all
from ..core.models import (
    EXTRACTABLE_STATES,
    MCP_VISIBLE_STATES,
    State,
)
from ..core.vault import init_vault, open_vault
from ..core.workflow import hard_delete, transition_state
from ..export import PROFILES, export_workbook, list_profiles
from ..extract import (
    chunk_all_eligible,
    chunk_conversation,
    extract_all_eligible,
    extract_for_conversation,
)
from ..ingest import import_path
from ..llm import LLMRunner, load_config as load_llm_config
from ..llm.chunking import llm_chunk_all_eligible, llm_chunk_conversation
from ..llm.label_groups import label_all_groups
from ..llm.summarize import summarize_all_eligible, summarize_conversation
from ..mcp import serve as mcp_serve
from ..search import (
    list_decisions,
    list_open_loops,
    project_view,
    search_chunks,
    search_conversations,
)
from ..search.search import list_projects
from ..store import open_store


def list_export_profiles() -> list[str]:
    return list_profiles()


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*[str(c) for c in r]))


def _iso(ts):
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


# ---------------------------------------------------------------------------

def cmd_init(args) -> int:
    v = init_vault(args.vault)
    print(f"Initialized vault at {v.root}")
    return 0


def cmd_import(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        result = import_path(
            vault, store, args.path, source=args.source, copy_raw=not args.no_copy
        )
    finally:
        store.close()
    print(f"Imported:                 {len(result.imported)}")
    print(f"Already present (dedup):  {len(result.deduped)}")
    print(f"Empty conversations:      {len(result.empty_skipped)}")
    print(f"Failed:                   {len(result.failed)}")
    if result.by_source:
        per_src = ", ".join(f"{k}={v}" for k, v in sorted(result.by_source.items()))
        print(f"By source:                {per_src}")
    print(f"Pending review now:       {result.pending_review_count_after}")
    if result.failed:
        print("\nFailures:")
        for title, err in result.failed[:10]:
            print(f"  - {title}: {err}")
    if result.empty_skipped and len(result.empty_skipped) <= 10:
        print("\nEmpty conversations skipped:")
        for t in result.empty_skipped:
            print(f"  - {t}")
    if result.raw_path:
        print(f"\nRaw archive copied to: {result.raw_path}")
    print("\nAll imported conversations are in 'pending_review'. Use:")
    print(f"  threadatlas review {args.vault}")
    return 0


def cmd_review(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        rows = store.list_conversations(state=args.state, limit=args.limit)
        if not rows:
            print(f"No conversations in state '{args.state}'.")
            return 0
        out = []
        for c in rows:
            title = c.title if len(c.title) <= 60 else c.title[:57] + "..."
            out.append([c.conversation_id, c.source, c.message_count, _iso(c.updated_at or c.created_at), title])
        _print_table(["id", "source", "msgs", "updated", "title"], out)
        print(f"\n{len(rows)} conversation(s) shown.")
    finally:
        store.close()
    return 0


def make_state_handler(target_state: str):
    def _handler(args) -> int:
        vault = open_vault(args.vault)
        store = open_store(vault)
        try:
            failures = []
            for cid in args.conversation_ids:
                try:
                    new_state = transition_state(store, cid, target_state)
                    print(f"{cid} -> {new_state}")
                except (KeyError, ValueError) as e:
                    failures.append((cid, str(e)))
            if failures:
                print("\nFailures:", file=sys.stderr)
                for cid, e in failures:
                    print(f"  {cid}: {e}", file=sys.stderr)
                return 1
        finally:
            store.close()
        return 0
    return _handler


def cmd_delete(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        if not args.yes:
            print("Hard delete is irreversible. Targets:")
            for cid in args.conversation_ids:
                c = store.get_conversation(cid)
                if c:
                    print(f"  {cid}: {c.title}")
                else:
                    print(f"  {cid}: <unknown>")
            ans = input("Proceed? [y/N] ").strip().lower()
            if ans != "y":
                print("Aborted.")
                return 1
        for cid in args.conversation_ids:
            try:
                report = hard_delete(vault, store, cid)
                print(json.dumps(report, indent=2))
            except KeyError as e:
                print(f"Skipped: {e}", file=sys.stderr)
    finally:
        store.close()
    return 0


def cmd_chunk(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        if args.conversation_id:
            chunks = chunk_conversation(store, args.conversation_id)
            print(f"{args.conversation_id}: {len(chunks)} chunks")
        else:
            res = chunk_all_eligible(store)
            total_chunks = sum(res.values())
            print(f"Chunked {len(res)} conversations, {total_chunks} chunks total.")
    finally:
        store.close()
    return 0


def cmd_extract(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        if args.conversation_id:
            counts = extract_for_conversation(store, args.conversation_id)
            print(json.dumps(counts, indent=2))
        else:
            res = extract_all_eligible(store)
            agg: dict[str, int] = {}
            for c in res.values():
                for k, n in c.items():
                    agg[k] = agg.get(k, 0) + n
            print(f"Extracted across {len(res)} conversations:")
            for k, v in sorted(agg.items()):
                print(f"  {k}: {v}")
    finally:
        store.close()
    return 0


def cmd_rebuild_index(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        store.rebuild_all_fts()
        store.conn.commit()
        print("FTS indexes rebuilt.")
    finally:
        store.close()
    return 0


def cmd_search(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        if args.include_private:
            visible = (State.INDEXED.value, State.PRIVATE.value)
        else:
            visible = tuple(MCP_VISIBLE_STATES)
        if args.mode == "conversations":
            hits = search_conversations(store, args.query, visible_states=visible, limit=args.limit)
        else:
            hits = search_chunks(store, args.query, visible_states=visible, limit=args.limit)
        if not hits:
            print("No results.")
            return 0
        rows = []
        for h in hits:
            snippet = (h.snippet or "").replace("\n", " ")
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            rows.append([h.conversation_id[:18], f"{h.score:.2f}", h.state, h.title[:40], snippet])
        _print_table(["conv_id", "score", "state", "title", "snippet"], rows)
    finally:
        store.close()
    return 0


def cmd_inspect(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        cid = args.conversation_id
        c = store.get_conversation(cid)
        if c is None:
            print(f"Unknown conversation: {cid}", file=sys.stderr)
            return 1
        msgs = store.list_messages(cid)
        chunks = store.list_chunks(cid)
        prov = store.list_provenance_for_conversation(cid)
        normalized_path = vault.normalized_path_for(cid)
        print(json.dumps({
            "conversation_id": c.conversation_id,
            "title": c.title,
            "source": c.source,
            "state": c.state,
            "imported_at": _iso(c.imported_at),
            "created_at": _iso(c.created_at),
            "updated_at": _iso(c.updated_at),
            "message_count_db": len(msgs),
            "chunk_count_db": len(chunks),
            "provenance_link_count_db": len(prov),
            "summary_short": c.summary_short,
            "manual_tags": c.manual_tags,
            "auto_tags": c.auto_tags,
            "importance_score": c.importance_score,
            "resurfacing_score": c.resurfacing_score,
            "has_open_loops": c.has_open_loops,
            "normalized_file_present": normalized_path.exists(),
            "normalized_file_path": str(normalized_path),
            "raw_imports_dir": str(vault.raw_imports),
            "mcp_visible": c.state in MCP_VISIBLE_STATES,
        }, indent=2))
    finally:
        store.close()
    return 0


def cmd_list_projects(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        if args.include_private:
            visible = (State.INDEXED.value, State.PRIVATE.value)
        else:
            visible = tuple(MCP_VISIBLE_STATES)
        rows = list_projects(store, visible_states=visible, limit=200)
        if not rows:
            print("No projects.")
            return 0
        _print_table(
            ["id", "title", "description"],
            [[r["object_id"][:22], r["title"][:30], (r["description"] or "")[:60]] for r in rows],
        )
    finally:
        store.close()
    return 0


def cmd_project(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        if args.include_private:
            visible = (State.INDEXED.value, State.PRIVATE.value)
        else:
            visible = tuple(MCP_VISIBLE_STATES)
        view = project_view(store, args.project_id, visible_states=visible)
        if view is None:
            print(f"No project visible with id {args.project_id}", file=sys.stderr)
            return 1
        print(json.dumps(view, indent=2, default=str))
    finally:
        store.close()
    return 0


def cmd_export(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        out = export_workbook(vault, store, profile=args.profile, out_path=args.out)
        print(f"Wrote {out}")
    finally:
        store.close()
    return 0


def cmd_mcp(args) -> int:
    return mcp_serve(args.vault)


def cmd_audit_conversation(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        data = audit_conversation(vault, store, args.conversation_id)
        if data is None:
            print(f"Unknown conversation: {args.conversation_id}", file=sys.stderr)
            return 1
        print(json.dumps(data, indent=2, default=str))
    finally:
        store.close()
    return 0


def cmd_audit_object(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        data = audit_object(store, args.object_id)
        if data is None:
            print(f"Unknown object: {args.object_id}", file=sys.stderr)
            return 1
        print(json.dumps(data, indent=2, default=str))
    finally:
        store.close()
    return 0


def cmd_plan_delete(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        for cid in args.conversation_ids:
            data = plan_hard_delete(vault, store, cid)
            if data is None:
                print(f"Unknown conversation: {cid}", file=sys.stderr)
                continue
            print(json.dumps(data, indent=2, default=str))
    finally:
        store.close()
    return 0


def _require_llm_runner(vault, task: str):
    """Load the LLM config and construct a runner, or raise with a clear error."""
    cfg = load_llm_config(vault.root)
    if cfg is None:
        raise SystemExit(
            f"LLM is not configured. Create {vault.root}/local_llm.json "
            f"with a 'command' and 'used_for' whitelist that includes '{task}'."
        )
    if not cfg.is_enabled_for(task):
        raise SystemExit(
            f"LLM task {task!r} is not enabled in local_llm.json 'used_for'. "
            f"Currently: {sorted(cfg.used_for)}"
        )
    return LLMRunner(vault, cfg)


def cmd_group(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        result = regroup_all(
            store,
            broad_k=args.broad,
            fine_k=args.fine,
            seed=args.seed,
        )
        if result.skipped_empty_corpus:
            print("Skipped: corpus too small to cluster.")
            return 0
        print(f"Members:    {result.members}")
        print(f"Broad k:    {result.broad_groups}")
        print(f"Fine k:     {result.fine_groups}")
        if args.llm_names:
            runner = _require_llm_runner(vault, "group_naming")
            outcomes = label_all_groups(vault, store, runner, level=None)
            ok = sum(1 for o in outcomes if o.success)
            fail = len(outcomes) - ok
            print(f"\nLLM-named groups: {ok} succeeded, {fail} skipped/failed")
    finally:
        store.close()
    return 0


def cmd_list_groups(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        groups = store.list_groups(level=args.level)
        if not groups:
            print("No groups. Run `threadatlas group <vault>` first.")
            return 0
        _print_table(
            ["id", "level", "n", "keyword_label", "llm_label"],
            [
                [g["group_id"][:22], g["level"], g["member_count"],
                 (g["keyword_label"] or "")[:40],
                 (g["llm_label"] or "")[:40]]
                for g in groups
            ],
        )
    finally:
        store.close()
    return 0


def cmd_group_view(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        g = store.get_group(args.group_id)
        if g is None:
            print(f"Unknown group: {args.group_id}", file=sys.stderr)
            return 1
        members = store.list_group_members(args.group_id)
        print(json.dumps({
            "group": g,
            "members": members,
        }, indent=2, default=str))
    finally:
        store.close()
    return 0


def cmd_summarize(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        runner = _require_llm_runner(vault, "summaries")
        if args.conversation_id:
            outcome = summarize_conversation(vault, store, runner, args.conversation_id)
            print(json.dumps(outcome.__dict__, indent=2))
        else:
            outcomes = summarize_all_eligible(vault, store, runner, limit=args.limit)
            ok = sum(1 for o in outcomes if o.success)
            print(f"Summarized: {ok} / {len(outcomes)}")
            for o in outcomes:
                if not o.success:
                    print(f"  fail: {o.conversation_id}: {o.error}")
    finally:
        store.close()
    return 0


def cmd_llm_chunk(args) -> int:
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        runner = _require_llm_runner(vault, "chunk_gating")
        if args.conversation_id:
            outcome = llm_chunk_conversation(vault, store, runner, args.conversation_id)
            print(json.dumps(outcome.__dict__, indent=2, default=str))
        else:
            outcomes = llm_chunk_all_eligible(vault, store, runner)
            merged_total = sum(o.merges for o in outcomes)
            print(f"Conversations: {len(outcomes)}")
            print(f"Merges applied: {merged_total}")
            llm_fail = sum(o.llm_failures for o in outcomes)
            if llm_fail:
                print(f"LLM failures: {llm_fail} (deterministic boundaries preserved)")
    finally:
        store.close()
    return 0


def cmd_process_approved(args) -> int:
    """Run chunk + extract across all indexed/private conversations.

    Idempotent and safe. Intended as a 'finish the post-approval pipeline'
    convenience; does NOT change state or approve anything on its own.
    """
    vault = open_vault(args.vault)
    store = open_store(vault)
    try:
        chunked = chunk_all_eligible(store)
        extracted = extract_all_eligible(store)
        print(f"Chunked: {len(chunked)} conversations, {sum(chunked.values())} chunks")
        print(f"Extracted: across {len(extracted)} conversations")
        agg: dict[str, int] = {}
        for c in extracted.values():
            for k, n in c.items():
                agg[k] = agg.get(k, 0) + n
        for k, v in sorted(agg.items()):
            print(f"  {k}: {v}")
    finally:
        store.close()
    return 0
