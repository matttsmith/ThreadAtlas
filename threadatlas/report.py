"""Static HTML report generator.

``threadatlas report ./vault`` writes a single self-contained HTML file
that an operator can open in any browser. There is no HTTP server and no
external fetch - CSS is inline; there are no scripts, fonts, or images.

The report is read-only: every mutation (approve, delete, tag, etc.)
still happens through the CLI. The report exists to make *triage* fast
and *auditability* visual, not to replace the explicit operator control
surface.

Sections:
* vault summary and state histogram
* recent imports (pending_review queue)
* groups at both levels with member counts (MCP-safe labels only)
* top projects, decisions, open loops, entities
* warnings (vault health issues the ``check`` command would flag)
"""

from __future__ import annotations

import html
import time
from datetime import datetime, timezone
from pathlib import Path

from .core.vault import Vault
from .store import Store


def _escape(s) -> str:
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


def _iso(ts):
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


# Minimal, boring CSS. No external fonts, no JS.
_CSS = """
html { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color: #1a1a1a; background: #fafafa; }
body { max-width: 1100px; margin: 24px auto; padding: 0 18px; line-height: 1.45; }
h1, h2, h3 { color: #111; }
h1 { border-bottom: 2px solid #222; padding-bottom: 4px; }
h2 { margin-top: 32px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #e4e4e4; font-size: 14px; }
th { background: #eee; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 10px 0; }
.card { border: 1px solid #ddd; background: #fff; padding: 10px 14px; border-radius: 6px; }
.card .val { font-size: 22px; font-weight: 600; }
.card .lbl { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: .04em; }
code { background: #eee; padding: 1px 4px; border-radius: 3px; font-size: 13px; }
.warn { background: #fff5e6; border: 1px solid #e0a966; padding: 8px 12px; border-radius: 6px; }
.muted { color: #666; }
small.meta { color: #888; }
.pill { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 11px; background: #eee; }
.pill-indexed { background: #d9eccf; color: #244f16; }
.pill-private { background: #eed9ee; color: #5c185c; }
.pill-pending_review { background: #f7efcf; color: #5a4a12; }
.pill-quarantined { background: #f2d7d7; color: #772020; }
"""


def _state_pill(state: str) -> str:
    cls = f"pill pill-{state}"
    return f'<span class="{_escape(cls)}">{_escape(state)}</span>'


def _table(headers: list[str], rows: list[list[str]], *, numeric_cols: set[int] | None = None) -> str:
    numeric_cols = numeric_cols or set()
    out = ["<table>"]
    out.append("<tr>" + "".join(f"<th>{_escape(h)}</th>" for h in headers) + "</tr>")
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            cls = ' class="num"' if i in numeric_cols else ""
            cells.append(f"<td{cls}>{v}</td>")  # v may be pre-escaped HTML
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def generate_report(vault: Vault, store: Store, out_path: Path | None = None) -> Path:
    """Generate the static HTML report."""
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = vault.root / "reports" / f"report_{ts}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- state histogram ----
    states_rows = store.conn.execute(
        "SELECT state, COUNT(*) AS c FROM conversations GROUP BY state ORDER BY state"
    ).fetchall()
    state_counts: dict[str, int] = {r["state"]: r["c"] for r in states_rows}
    total = sum(state_counts.values())

    # ---- source counts ----
    src_rows = store.conn.execute(
        "SELECT source, COUNT(*) AS c FROM conversations GROUP BY source"
    ).fetchall()

    # ---- recent pending_review ----
    pending_rows = store.conn.execute(
        """
        SELECT conversation_id, source, title, message_count, imported_at
          FROM conversations
         WHERE state = 'pending_review'
         ORDER BY imported_at DESC
         LIMIT 50
        """
    ).fetchall()

    # ---- derived object counts ----
    derived_rows = store.conn.execute(
        """
        SELECT kind, COUNT(DISTINCT o.object_id) AS c
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE c.state = 'indexed' AND o.state = 'active'
         GROUP BY kind
        """
    ).fetchall()
    derived_counts = {r["kind"]: r["c"] for r in derived_rows}

    # ---- groups ----
    groups_rows = store.conn.execute(
        """
        SELECT group_id, level, keyword_label, llm_label, member_count
          FROM conversation_groups
         ORDER BY level, member_count DESC
         LIMIT 200
        """
    ).fetchall()

    # ---- top open loops / decisions ----
    top_loops = store.conn.execute(
        """
        SELECT o.object_id, o.title,
               COUNT(DISTINCT p.conversation_id) AS convs
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE o.kind = 'open_loop' AND o.state = 'active' AND c.state = 'indexed'
         GROUP BY o.object_id
         ORDER BY convs DESC, o.title
         LIMIT 20
        """
    ).fetchall()

    top_decisions = store.conn.execute(
        """
        SELECT o.object_id, o.title,
               COUNT(DISTINCT p.conversation_id) AS convs
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
          JOIN conversations c ON c.conversation_id = p.conversation_id
         WHERE o.kind = 'decision' AND o.state = 'active' AND c.state = 'indexed'
         GROUP BY o.object_id
         ORDER BY convs DESC, o.title
         LIMIT 20
        """
    ).fetchall()

    # ---- warnings ----
    from . import health  # local import; health module lives next to this one
    warnings = health.quick_check(vault, store)

    # ---- compose HTML ----
    html_parts: list[str] = []
    html_parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    html_parts.append("<meta name='robots' content='noindex,nofollow,noarchive'>")
    html_parts.append("<title>ThreadAtlas report</title>")
    html_parts.append(f"<style>{_CSS}</style></head><body>")
    html_parts.append(f"<h1>ThreadAtlas report</h1>")
    html_parts.append(f"<p class='muted'>Vault: <code>{_escape(vault.root)}</code> &middot; generated {_escape(_iso(time.time()))} UTC</p>")

    # Summary cards.
    def _card(val, lbl):
        return f"<div class='card'><div class='val'>{_escape(val)}</div><div class='lbl'>{_escape(lbl)}</div></div>"

    html_parts.append("<div class='grid'>")
    html_parts.append(_card(total, "total conversations"))
    for s in ["indexed", "private", "pending_review", "quarantined"]:
        html_parts.append(_card(state_counts.get(s, 0), s))
    html_parts.append("</div>")

    # Source mix.
    html_parts.append("<h2>Source mix</h2>")
    html_parts.append(_table(
        ["source", "count"],
        [[_escape(r["source"]), f"{r['c']}"] for r in src_rows],
        numeric_cols={1},
    ))

    # Warnings.
    if warnings:
        html_parts.append("<h2>Warnings</h2>")
        for w in warnings:
            html_parts.append(f"<div class='warn'>{_escape(w)}</div>")

    # Pending review.
    html_parts.append(f"<h2>Pending review <small class='meta'>(top 50)</small></h2>")
    if not pending_rows:
        html_parts.append("<p class='muted'>No conversations awaiting review.</p>")
    else:
        html_parts.append(_table(
            ["id", "source", "imported", "msgs", "title"],
            [
                [
                    f"<code>{_escape(r['conversation_id'])}</code>",
                    _escape(r["source"]),
                    _escape(_iso(r["imported_at"])),
                    str(r["message_count"] or 0),
                    _escape(r["title"]),
                ]
                for r in pending_rows
            ],
            numeric_cols={3},
        ))

    # Derived objects summary.
    html_parts.append("<h2>Indexed corpus at a glance</h2>")
    html_parts.append("<div class='grid'>")
    for kind in ("project", "decision", "open_loop", "entity", "preference", "artifact"):
        html_parts.append(_card(derived_counts.get(kind, 0), kind))
    html_parts.append("</div>")

    # Top open loops.
    html_parts.append("<h2>Top open loops</h2>")
    if not top_loops:
        html_parts.append("<p class='muted'>None yet. Run <code>threadatlas extract</code>.</p>")
    else:
        html_parts.append(_table(
            ["object_id", "title", "convs"],
            [
                [f"<code>{_escape(r['object_id'])}</code>", _escape(r["title"]), str(r["convs"])]
                for r in top_loops
            ],
            numeric_cols={2},
        ))

    # Top decisions.
    html_parts.append("<h2>Top decisions</h2>")
    if not top_decisions:
        html_parts.append("<p class='muted'>None yet.</p>")
    else:
        html_parts.append(_table(
            ["object_id", "title", "convs"],
            [
                [f"<code>{_escape(r['object_id'])}</code>", _escape(r["title"]), str(r["convs"])]
                for r in top_decisions
            ],
            numeric_cols={2},
        ))

    # Groups - both levels.
    html_parts.append("<h2>Groups</h2>")
    if not groups_rows:
        html_parts.append("<p class='muted'>No groups yet. Run <code>threadatlas group</code>.</p>")
    else:
        broad = [r for r in groups_rows if r["level"] == "broad"]
        fine = [r for r in groups_rows if r["level"] == "fine"]

        def _group_rows(rows):
            return [
                [
                    f"<code>{_escape(r['group_id'])}</code>",
                    str(r["member_count"]),
                    _escape(r["llm_label"] or ""),
                    _escape(r["keyword_label"] or ""),
                ]
                for r in rows
            ]

        if broad:
            html_parts.append(f"<h3>Broad ({len(broad)})</h3>")
            html_parts.append(_table(
                ["id", "members", "llm label", "keyword label"],
                _group_rows(broad),
                numeric_cols={1},
            ))
        if fine:
            html_parts.append(f"<h3>Fine ({len(fine)})</h3>")
            html_parts.append(_table(
                ["id", "members", "llm label", "keyword label"],
                _group_rows(fine),
                numeric_cols={1},
            ))

    html_parts.append("</body></html>")
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    return out_path
