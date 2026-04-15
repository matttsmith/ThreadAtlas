# ThreadAtlas operator guide

This is the short, practical guide for the one person who actually runs
ThreadAtlas on their laptop. For architecture, see `docs/spec.md`.

## The mental model

ThreadAtlas is a **local filing cabinet** for AI chat history. It is not a
search engine, not a SaaS app, and not an agent. You import exports, review
them, approve the ones you want to keep, and either search them yourself or
expose the approved subset to Claude via MCP.

The source of truth is the vault:

```
vault/
  raw_imports/    # your original export archives (kept for provenance)
  normalized/     # canonical per-conversation JSON - human-readable
  db/             # SQLite index + FTS + derived-object metadata
  exports/        # generated XLSX workbooks
  cache/          # ephemeral; always safe to delete
  logs/           # local only; no conversation content
```

If the DB is ever corrupted, you can rebuild it from `raw_imports/`.

## Visibility states — the one table to keep in your head

| State            | In CLI search (default) | In CLI search `--include-private` | Visible to MCP | In synthesis / project pages | Has FTS rows | Has chunks + derived objects |
|------------------|:-----------------------:|:---------------------------------:|:--------------:|:----------------------------:|:------------:|:----------------------------:|
| `pending_review` | no                      | no                                | no             | no                           | no           | no                           |
| `indexed`        | yes                     | yes                               | **yes**        | yes                          | yes          | yes                          |
| `private`        | no                      | yes                               | no             | no                           | yes          | yes                          |
| `quarantined`    | no                      | no                                | no             | no                           | no           | no                           |

After hard delete the conversation does not exist anywhere in ThreadAtlas.

### Which state for what

- **`pending_review`** — the default on import. Nothing has been indexed yet;
  nothing is MCP-visible. Use this to triage.
- **`indexed`** — the conversation is treated as approved working knowledge.
  Claude can see it via MCP.
- **`private`** — you want it searchable from the CLI but Claude should never
  see it. Use this for sensitive professional threads and personal content.
- **`quarantined`** — you're unsure and want it out of every index but don't
  want to delete it yet. The normalized JSON is kept; chunks, FTS rows, and
  provenance are stripped.

## Happy path

```
threadatlas init ./vault
threadatlas import ./vault /path/to/chatgpt-export.zip --source chatgpt
threadatlas import ./vault /path/to/claude-export.zip --source claude
threadatlas review ./vault

# Approve the work threads, mark personal threads private, quarantine
# unsure ones, delete obvious noise.
threadatlas approve     ./vault conv_1 conv_2 conv_3
threadatlas private     ./vault conv_personal_1
threadatlas quarantine  ./vault conv_unsure_1
threadatlas delete      ./vault conv_garbage_1

# Finish the post-approval pipeline (chunks + heuristic extraction).
threadatlas process-approved ./vault

# Day-to-day use.
threadatlas search ./vault "project atlas staffing"
threadatlas list-projects ./vault
threadatlas export ./vault --profile review_workbook
threadatlas mcp ./vault   # connect Claude to approved content
```

## Deletion — what it actually does

`threadatlas delete` is a **hard delete** and it is irreversible. It removes:

- the conversation row (SQLite)
- all messages (FK cascade)
- all chunks (FK cascade)
- all provenance links (FK cascade)
- all FTS5 rows for the conversation
- any derived objects (projects, decisions, etc.) whose only provenance was
  this conversation; shared objects survive with reduced provenance
- the normalized JSON file on disk
- runs `VACUUM` on the DB so freed pages don't linger

Before doing it, **preview with `threadatlas plan-delete <id>`** to see
exactly what will be removed, which objects would become orphans, and which
would survive.

`threadatlas delete` with no `--yes` flag prompts for confirmation.

The raw archive in `vault/raw_imports/` is NOT removed; if you want it gone,
delete it by hand. This is intentional — raw imports are provenance, and
removing them could silently make other conversations un-reproducible.

## Audit — answering "why is this here?"

```
threadatlas audit-conversation ./vault conv_abc
threadatlas audit-object ./vault obj_xyz
```

- **`audit-conversation`** prints a full dump: metadata, chunk list, every
  derived object this conversation contributes to, and a sample of
  provenance excerpts.
- **`audit-object`** prints a derived object plus every provenance link:
  source conversation title + state, chunk id, excerpt. Good for deciding
  whether an object is real or a false positive.

## MCP — what Claude sees

Start with `threadatlas mcp ./vault`. The server runs over stdio; there is
no TCP listener. Every tool filters to `indexed` state only. `private` and
`quarantined` are invisible. A dedicated audit tool
(`inspect_conversation_storage`) always returns metadata counts but redacts
the title for non-visible conversations, so you can use Claude to help audit
your own storage without leaking titles.

## XLSX — what's in which workbook

| Profile              | States included                                   | Sheets                                                                                     |
|----------------------|---------------------------------------------------|--------------------------------------------------------------------------------------------|
| `conversations_only` | pending_review, indexed, private, quarantined     | conversations                                                                              |
| `review_workbook`    | pending_review, indexed, private                  | conversations, chunks                                                                      |
| `project_workbook`   | **indexed only**                                  | conversations, chunks, projects, decisions, open_loops, entities, provenance               |
| `full_analysis`      | pending_review, indexed, private, quarantined     | conversations, chunks, projects, decisions, open_loops, entities, preferences, artifacts, provenance |

Column names are stable; the test suite fails if they drift.

## What v1 intentionally does NOT do

- **No cloud APIs.** No OpenAI, no Gemini, no Anthropic API. No outbound
  connections at all.
- **No local LLM calls by default.** Summaries and extraction are
  deterministic regex/heuristic. If you later add a local LLM, do it via
  a stdio subprocess; nothing else.
- **No auto-merge of projects or entities.** Under-merge is the default;
  you can merge by hand later.
- **No background workers.** Every action is an explicit CLI command.
- **No TUI, no progress bars, no ncurses.** The output is designed to be
  piped into other tools.
- **No in-app updater.**

## Troubleshooting

- **"sqlite3.OperationalError: cannot DELETE from contentless fts5 table"** —
  you're on an old version. Upgrade.
- **`threadatlas import` reports "failed"** — the parser hit a malformed
  conversation. Other conversations in the same archive still imported. The
  error string is included in the report.
- **Search is missing a conversation you just approved** — run
  `threadatlas process-approved ./vault` (or `threadatlas rebuild-index`).
  Approve only writes the state; it doesn't re-run chunk/extract.
- **"Disallowed transition: X -> Y"** — the state whitelist refused. Use
  `pending_review` as an intermediate state, or just re-approve from whichever
  state you're in.
- **DB corrupted** — you can rebuild from `vault/normalized/` JSON files.
  The DB is just an index; the JSON is the source of truth.
