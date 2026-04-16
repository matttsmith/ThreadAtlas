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

## Thematic grouping

`threadatlas group ./vault` clusters your approved corpus into two levels
of buckets:

- **Broad** (default k=10) — coarse topic buckets.
- **Fine** (default k=100) — fine-grained groupings.

The clustering itself is **deterministic**: TF-IDF over conversation
titles + summaries, L2-normalized sparse vectors, k-means++ with fixed seed.
Same corpus + same seed + same k → identical groups.

Group labels come in two flavors:

- **Keyword label** — always present. Top distinctive terms per cluster,
  e.g. `chs, staffing, q2, budget, planning`.
- **LLM label** — optional. Prose name like `CHS program management`.
  Only appears if you configured a local LLM (see below) and passed
  `--llm-names`.

```
threadatlas group ./vault                      # deterministic only
threadatlas group ./vault --broad 8 --fine 80  # tune k's
threadatlas group ./vault --llm-names          # prose names (requires local LLM)
threadatlas list-groups ./vault
threadatlas group-view ./vault grp_xyz
```

Both labels are exported in the `conversations` sheet
(`broad_group_label`, `fine_group_label`). MCP exposes `list_groups` and
`get_group` over indexed-only members.

Groups are **wiped and rebuilt** each time you run `threadatlas group`;
there is no incremental update. Run after important imports or after
you've changed the state of a meaningful slice of conversations.

## Optional: local LLM integration

ThreadAtlas never talks to the network by default. If you want
higher-quality summaries, prose group names, or LLM-assisted chunk
refinement, you run a **local** model yourself and point ThreadAtlas
at it via one of two backends.

### Backend A: subprocess (zero-network)

Invoke a local binary (llama.cpp CLI, MLX-LM, llamafile) directly.
Create `<vault>/local_llm.json`:

```json
{
  "command": [
    "/usr/local/bin/llama-cli",
    "-m", "/path/to/qwen2.5-3b-instruct-q4.gguf",
    "--prompt-file", "{PROMPT_FILE}",
    "--no-conversation",
    "--temp", "0.1",
    "--n-predict", "256"
  ],
  "timeout_seconds": 120,
  "max_prompt_chars": 12000,
  "max_response_chars": 4000,
  "used_for": ["summaries", "group_naming", "chunk_gating"],
  "dry_run": false
}
```

- `command` is the argv. `{PROMPT_FILE}` is substituted with a temp
  file; `{PROMPT}` is inline substitution; if neither appears, the
  prompt goes on stdin.

### Backend B: llama-server (loopback HTTP)

Talk to a locally-running llama-server (or any OpenAI-compatible
endpoint) over HTTP.  Start `llama-server` yourself, then configure:

```json
{
  "provider": "llama_server",
  "base_url": "http://127.0.0.1:8080",
  "model": "qwen2.5-3b-instruct",
  "temperature": 0.1,
  "max_tokens": 256,
  "timeout_seconds": 120,
  "max_prompt_chars": 12000,
  "max_response_chars": 4000,
  "used_for": ["summaries", "group_naming", "chunk_gating"],
  "dry_run": false
}
```

- `base_url` must point to a **loopback** address (127.0.0.1 /
  localhost / ::1).  Non-loopback URLs are rejected unless you set
  `"allow_nonlocal_host": true` — which you should not do unless you
  understand the privacy implications.
- `model` is the model alias as reported by `/v1/models`. If omitted,
  the server's default is used.
- `healthcheck_on_start` (default true): the runner will verify
  `/v1/models` before the first call.

### Common fields (both backends)

- `used_for` is a **whitelist**. A task not listed will be refused.
  Default shipping state is the feature disabled.
- `dry_run: true` makes every call print the prompt locally and skip
  the backend. Use this to inspect exactly what would be sent.
- Every call is logged (metadata only, no content) to
  `<vault>/logs/llm_calls.jsonl`.
- `timeout_seconds`, `max_prompt_chars`, `max_response_chars` — caps
  enforced per-call across both backends.

Hardware guidance for a MacBook Air M4 with 24 GB unified memory:
Qwen2.5-3B-Instruct Q4 (~2 GB) or Llama-3.2-3B-Instruct Q4 (~2 GB)
are more than enough for summarization and group naming. Larger
models (7B) give marginally better prose for group names.

### LLM commands

- `threadatlas llm-check ./vault` — validate config (subprocess:
  executable exists; llama_server: `/v1/models` reachable).
- `threadatlas llm-check ./vault --probe` — also send a tiny
  completion to confirm the model responds.
- `threadatlas summarize ./vault` — generate topical 2-3 sentence
  summaries; updates `summary_short` and sets `summary_source = llm`.
- `threadatlas summarize ./vault --conversation-id conv_xxx` — just one.
- `threadatlas group ./vault --llm-names` — prose names for each cluster.
- `threadatlas llm-chunk ./vault` — refine chunk boundaries (see next).

### LLM-assisted chunking

The deterministic chunker proposes boundaries. `threadatlas llm-chunk`
asks the LLM, for each adjacent boundary, "is this a clear topic
shift?" — and **only** removes boundaries the LLM says aren't clear.
The LLM cannot add splits, only collapse them. Net effect: fewer,
more defensible chunks.

The prompt biases the LLM toward "not a split" when uncertain, so the
failure mode is "keep existing chunks" rather than "hallucinate new
ones." On malformed LLM output, the original deterministic boundary
is preserved.

## What v1 intentionally does NOT do

- **No cloud APIs.** No OpenAI, no Gemini, no Anthropic API. No outbound
  connections at all.
- **No local LLM by default.** You must install a model and create
  `local_llm.json` yourself. Nothing is downloaded.
- **No auto-merge of projects or entities.** Under-merge is the default;
  you can merge by hand later.
- **No background workers.** Every action is an explicit CLI command.
- **No progress bars or spinners.** CLI output is designed to be piped
  into other tools. The optional curses TUI (`threadatlas tui`) is
  read-only and takes over its own terminal; it is never in the pipe.
- **No in-app updater.**
- **LLM cannot invent new chunk splits.** The boundary gate can only
  remove, never add.
- **LLM is never used for decisions / open loops / entity extraction.**
  Those stay deterministic.

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
