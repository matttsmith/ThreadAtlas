# ThreadAtlas

**Fully local conversation intelligence for your ChatGPT and Claude export
archives.** Ingest, review, search, cluster, and export your AI chat history
without a single outbound network request.

![tests](https://github.com/matttsmith/threadatlas/actions/workflows/test.yml/badge.svg)

---

## What it is

ThreadAtlas turns your exported AI chats into a searchable, trustworthy local
knowledge base. It:

- **Ingests** ChatGPT and Claude exports (zip, directory, or raw
  `conversations.json`), with a modular parser registry for new sources.
- **Partitions** conversations by sensitivity via an explicit review
  workflow: `pending_review`, `indexed`, `private`, `quarantined`, or hard
  `deleted`.
- **Searches** locally with hybrid keyword (FTS5) + semantic (TF-IDF
  embeddings) search, merged via reciprocal rank fusion. CLI + MCP
  surfaces respect the visibility rules.
- **Chunks** long threads into thematic segments using a deterministic
  heuristic, optionally refined by a local LLM (merge-only; the LLM can
  never introduce new splits).
- **Classifies** each message by register (work, research, roleplay,
  creative_writing, etc.) and reality_mode (literal, fictional,
  hypothetical) via a local LLM, so roleplay and jailbreak content is
  automatically separated from substantive work.
- **Extracts** projects, entities, decisions, open loops, artifacts, and
  preferences — via LLM when configured, or deterministic regex
  heuristics as fallback — each with provenance excerpts and type
  disambiguation.
- **Clusters** your corpus into coarse + fine thematic groups via
  pure-Python TF-IDF + k-means, with optional local-LLM naming.
- **Summarizes** conversations via an optional local LLM subprocess.
- **Exports** polished XLSX workbooks for manual triage.
- **Exposes** an audit surface over stdio MCP so Claude can reason across
  your approved corpus — and *only* the approved corpus.
- **Provides** an interactive ASCII TUI dashboard (stdlib curses) and a
  static HTML report for visual overview.

---

## Core principles

| Principle | What it means in practice |
|---|---|
| **Fully local** | No outbound network. A static test fails the build if any networking import lands in the runtime package. |
| **Review before exposure** | Every new import lands in `pending_review`. Nothing is MCP-visible or synthesized until you explicitly approve it. |
| **Privacy boundaries are product boundaries** | `private` content is locally searchable but hidden from MCP and from global synthesis. `quarantined` is stored raw with all derivative surfaces stripped. |
| **Hard delete is real** | `threadatlas delete` removes messages, chunks, FTS rows, derivative objects whose only provenance was this conversation, and the normalized JSON file. `VACUUM`'d after. |
| **Deterministic by default** | TF-IDF + k-means are fixed-seed. Extraction heuristics are inspectable regexes. The LLM is optional. |
| **Boring, auditable CLI** | Mutations need confirmation (`--yes` to bypass). The TUI is read-only. |
| **Spreadsheet is a first-class interface** | Workbook profiles are stable, sortable, filterable; column schemas are locked down by tests. |

ThreadAtlas is designed for a single technically-sophisticated user on their
own machine. It is **not** a SaaS, a server, a team product, or a
replacement for AI chat itself.

---

## Install

Python ≥ 3.10. Only runtime dependency is `openpyxl` (XLSX). The rest is
standard library.

```
git clone https://github.com/matttsmith/threadatlas
cd threadatlas
pip install -e ".[dev]"
```

On macOS, `curses` is built into Python. On Windows, install
`windows-curses` if you want the TUI.

---

## Quick start

```bash
# 1. Create a vault.
threadatlas init ./vault

# 2. Import exports (ChatGPT or Claude, zip / dir / conversations.json).
threadatlas import ./vault ~/Downloads/chatgpt-export.zip --source chatgpt
threadatlas import ./vault ~/Downloads/claude-export.zip  --source claude

# Optional: skip the pending_review step for non-sensitive threads.
# auto_rules.json (see below) still routes matching threads to private.
threadatlas import ./vault ./export --auto-approve

# 3. Review what landed in pending_review.
threadatlas review ./vault
threadatlas tui ./vault                    # interactive ASCII dashboard

# 4. Decide what to approve, keep private, quarantine, or delete.
threadatlas approve    ./vault <conv_id>   # becomes MCP-visible
threadatlas private    ./vault <conv_id>   # CLI-only, hidden from MCP
threadatlas quarantine ./vault <conv_id>   # normalized only, no derivatives
threadatlas delete     ./vault <conv_id>   # hard delete (confirmation required)

# 5. Build chunks + derived objects + thematic groups.
threadatlas process-approved ./vault       # chunk + extract in one go
threadatlas group ./vault                  # broad (10) + fine (100) clusters

# 6. Search.
threadatlas search ./vault "CHS staffing"
threadatlas search ./vault "therapy" --include-private

# 7. Export.
threadatlas export ./vault --profile review_workbook
threadatlas export ./vault --profile project_workbook
threadatlas report ./vault                 # static HTML dashboard

# 8. Let Claude query your approved corpus (see docs/mcp_setup.md).
threadatlas mcp ./vault                    # stdio MCP server

# 9. Operator hygiene.
threadatlas check ./vault                  # vault health invariants
threadatlas audit-conversation ./vault <conv_id>
threadatlas rebuild-from-normalized ./vault
```

Full CLI reference: `threadatlas --help` and `threadatlas <cmd> --help`.

### Auto-classification rules

The pending-review step is friction. For operators who will approve most
threads anyway, you can layer a keyword-/regex-based safety net that
auto-routes sensitive threads straight to `private` (or `quarantined`)
on import. Drop a `<vault>/auto_rules.json`:

```json
{
  "auto_private": [
    {"patterns": ["therapy", "therapist", "anxiety medication"],
     "fields": ["title", "messages"]},
    {"patterns": ["\\b\\d{3}-\\d{2}-\\d{4}\\b"],
     "mode": "regex", "fields": ["messages"]}
  ],
  "auto_quarantine": [
    {"patterns": ["[no-index]", "[secret]"], "fields": ["title"]}
  ]
}
```

- Rules **only down-classify.** A rule cannot make something more
  visible. A matching rule always wins over `--auto-approve`.
- `threadatlas rescan-rules ./vault` re-applies the current rules to the
  existing corpus (also down-classify-only).
- The matching pattern is recorded in each conversation's
  `notes_local` so you can see why it landed where it did.

### Wiring to Claude Desktop / other MCP hosts

See **[docs/mcp_setup.md](docs/mcp_setup.md)** for the exact
`claude_desktop_config.json` config, platform-specific paths, log
locations, verification, and troubleshooting. TL;DR:

```json
{
  "mcpServers": {
    "threadatlas": {
      "command": "/absolute/path/to/threadatlas",
      "args": ["mcp", "/absolute/path/to/vault"]
    }
  }
}
```

Restart Claude Desktop, start a new conversation, the ThreadAtlas tools
appear. Claude sees only `indexed` conversations; there are no mutating
tools.

---

## Optional: local LLM integration

ThreadAtlas never connects to the internet by default. If you want
higher-quality summaries, prose group names, or LLM-assisted chunk
refinement, install a local model and point ThreadAtlas at it via one of
two backends.

### Option A: subprocess (original, zero-network)

Invoke a local binary (llama.cpp CLI, MLX-LM, llamafile) directly:

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

### Option B: llama-server (loopback HTTP)

Talk to a locally-running llama-server (or any OpenAI-compatible
endpoint) over loopback. Start the server separately, then configure:

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
  "used_for": ["summaries", "group_naming", "chunk_gating",
               "turn_classification", "extraction", "profile"],
  "dry_run": false
}
```

Non-loopback URLs are rejected unless `"allow_nonlocal_host": true` is
explicitly set.

### Validate and use

```bash
threadatlas llm-check ./vault                    # validate config + server
threadatlas llm-check ./vault --probe            # also send a test completion
threadatlas summarize ./vault                    # 2-3 sentence summaries
threadatlas group ./vault --llm-names            # prose cluster names
threadatlas llm-chunk ./vault                    # refine chunk boundaries
```

### v2 LLM tasks

The v2 pipeline adds three new LLM task types (opt-in via `used_for`):

- **`turn_classification`**: Classifies each message by register (work,
  research, career_logistics, creative_writing, roleplay,
  jailbreak_experiment, puzzle_hobby, personal, other) and reality_mode
  (literal, fictional, hypothetical). This is the single highest-value
  change — it filters roleplay and fictional content from substantive
  extractions.
- **`extraction`**: Replaces regex heuristics with LLM-based extraction
  of projects, decisions, open loops, and entities. Respects the
  per-turn classifications to avoid extracting fictional commitments.
- **`profile`**: Generates a narrative user profile on demand, organized
  by topic (active projects, recent interests, dormant threads,
  recurring preoccupations, stylistic tendencies).

Prompts are stored in the `prompts/` directory as versioned text files.
Each extraction logs the prompt version used, enabling clean re-runs when
prompts improve.

### Safeguards

- **`used_for` is a whitelist.** A task not listed is refused. Default is
  LLM fully disabled.
- **Per-call timeout + prompt/response size caps.**
- **Every call logged to `vault/logs/llm_calls.jsonl` — metadata only, no
  prompt or response content.**
- **`dry_run: true`** makes every call print the prompt locally and skip
  the backend. Use this to audit exactly what would be sent.
- **Chunk boundary gate can only merge**, never introduce new splits. If
  the LLM fails or returns malformed JSON, the deterministic boundary is
  preserved.
- **Loopback-only by default.** The llama_server backend refuses
  non-127.0.0.1/localhost URLs unless explicitly overridden.

A 3B-class quantized Qwen or Llama model on an M-series Mac runs the
summarization workflow comfortably.

---

## Architecture

```
threadatlas/
  core/        domain models, vault layout, state machine, deletion cascade
  ingest/      parser registry (ChatGPT + Claude) + import pipeline
  store/       SQLite + FTS5 schema + normalized JSON IO
  extract/     deterministic chunking + heuristic derived objects
  search/      hybrid keyword+semantic search, embeddings, project synthesis
  cluster/     TF-IDF + k-means grouping (stdlib, deterministic)
  llm/         local-LLM: subprocess + llama_server backends, 2-pass pipeline, profile gen
  prompts/     versioned prompt files for LLM tasks (turn classification, extraction, profile)
  export/      XLSX workbook profiles
  mcp/         stdio JSON-RPC server (read-only, indexed-only)
  tui/         curses dashboard (read-only, responsive)
  cli/         argparse entry points
  audit.py     operator audit/inspection helpers
  health.py    vault invariant checker
  recovery.py  disaster recovery from normalized JSON
  report.py    static HTML report
```

The **normalized JSON files** in `vault/normalized/` are the recoverable
source of truth; SQLite is the index over them. You can lose or delete
the DB and rebuild it with `threadatlas rebuild-from-normalized`.

---

## Visibility matrix

| Surface | pending_review | indexed | private | quarantined |
|---|---|---|---|---|
| CLI `search` (default) | ❌ | ✅ | ❌ | ❌ |
| CLI `search --include-private` | ❌ | ✅ | ✅ | ❌ |
| MCP (all tools) | ❌ | ✅ | ❌ | ❌ |
| Project synthesis | ❌ | ✅ | ❌ | ❌ |
| Chunks + extraction | ❌ | ✅ | ✅ | ❌ |
| FTS rows exist | ❌ | ✅ | ✅ | ❌ |
| Normalized JSON on disk | ✅ | ✅ | ✅ | ✅ |
| Grouping participation | ❌ | ✅ | ✅ | ❌ |
| MCP group labels derived from | — | ✅ only | ❌ | — |

Group labels shown over MCP are **recomputed** from only the indexed
members of a group so private content cannot leak into a label that
Claude sees.

---

## MCP tool surface

All tools are read-only by default and expose only `indexed` conversations.

### Search & query

| Tool | Description |
|---|---|
| `query` | Structured hybrid search with filter prefixes: `source:`, `tag:`, `kind:`, `project:`, `after:`, `before:`, `has:`, `register:`. |
| `search_conversations` | Hybrid keyword + semantic search over conversations. |
| `search_chunks` | Keyword search over thematic chunks. |
| `find_related` | Semantic similarity search given free-text context. |

### Conversation detail

| Tool | Description |
|---|---|
| `get_conversation_summary` | LLM-generated summary + register classification + metadata. |
| `get_conversation_messages` | Raw messages for one conversation. |
| `get_conversation_chunks` | Thematic chunks for one conversation. |

### Structured data

| Tool | Description |
|---|---|
| `list_projects` | Active projects (auto-excludes roleplay/jailbreak). |
| `get_project` | Project page: linked conversations, decisions, open loops, entities. |
| `get_project_timeline` | Chronological timeline for a project. |
| `list_decisions` | User commitments from literal-mode conversations. |
| `list_open_loops` | Unresolved tasks/questions from literal-mode conversations. |
| `list_entities` | People, orgs, concepts, artifacts with type disambiguation. |
| `list_groups` | Thematic conversation clusters. |
| `get_group` | Group detail with indexed members. |

### Profile & audit

| Tool | Description |
|---|---|
| `generate_profile` | Narrative user profile organized by topic. Cached 7 days. Optional `focus` parameter. |
| `inspect_conversation_storage` | Metadata-only audit of stored data. |

### Common filter parameters

All `list_*` and `search_*` tools accept these optional filters:

| Parameter | Type | Description |
|---|---|---|
| `after` | ISO date string | Only include items after this date. |
| `before` | ISO date string | Only include items before this date. |
| `register` | array of strings | Filter by conversation register. Defaults exclude `roleplay` and `jailbreak_experiment`. |
| `source` | `chatgpt` \| `claude` \| `all` | Filter by conversation source. |

---

## Hard delete semantics

`threadatlas delete <vault> <conv_id>` physically removes:

- the conversation row and all its messages (FK cascade)
- all chunks
- all FTS5 rows
- every provenance link pointing at the conversation
- derived objects whose **only** provenance was this conversation
- the normalized JSON file on disk
- then runs `VACUUM` to release pages

Derived objects referenced by *other* conversations survive, with only
this conversation's contribution removed. Exports and reports are
generated on demand from the current DB, so next run no longer reflects
deleted content.

---

## Tests and CI

The suite has ~395 tests covering parsing, state transitions, visibility
boundaries, hard-delete cascade, chunking, heuristic extraction, search
ranking, XLSX schema stability, MCP tool behavior, LLM subprocess runner,
TF-IDF + k-means determinism, full LLM pipeline integration against a
smart fake, v2 LLM pipeline (turn classification, extraction, content
hashing), embedding generation and hybrid search, reciprocal rank fusion,
register/date filtering, new MCP tools (profile, find_related), prompt
loading and versioning, schema migration, and a static no-network
enforcement check.

```
pytest -q
```

CI runs the suite on Python 3.10, 3.11, and 3.12 on every push and PR.
A separate job runs only `tests/test_no_network.py` so the privacy
invariant is surfaced independently of general regressions.

### Running pipeline tests against a real local model

```bash
export THREADATLAS_REAL_LLM_ARGV='["/usr/local/bin/llama-cli",
    "-m", "/path/to/model.gguf",
    "--prompt-file", "{PROMPT_FILE}",
    "--no-conversation", "--temp", "0.1", "--n-predict", "256"]'
export THREADATLAS_REAL_LLM_TIMEOUT=120
pytest tests/test_llm_pipeline_integration.py -q
```

The assertions are lenient enough for a well-behaved 3B-class local model.
If your model fails them, that's a signal about the model.

---

## What ThreadAtlas does NOT do

- Talk to the internet. There is no optional connected mode.
- Send anything to a cloud API (OpenAI, Anthropic, Gemini). None.
- Auto-download models or embeddings.
- Run a background daemon, file watcher, or server.
- Expose a localhost HTTP dashboard (would add CSRF/XSS surface;
  static HTML report covers visual overview without a TCP listener).
- Auto-merge derived objects or projects (`obj-merge` is operator-driven).
- Import from email, Drive, Slack, or anything non-AI-chat.
- Sync between machines. Files on disk are the source of truth.

---

## Adapting or extending

**Add a new source parser** (e.g. Gemini exports): drop a new module in
`threadatlas/ingest/`, subclass `Parser`, implement `can_handle(path)` and
`iter_conversations(path)`, call `registry.register(MyParser())`. No
other layer needs changing — import pipeline, store, search, MCP, XLSX
all operate on the canonical `ParsedConversation`/`Conversation`/`Message`
shape.

**Add a derived-object kind**: add to `DerivedKind`, add a harvester in
`extract/heuristics.py`, add an XLSX sheet builder and a test. Keep the
precision bar high.

**Tweak the LLM prompts**: v2 prompts live in the `prompts/` directory as
versioned text files. Legacy prompts remain in `threadatlas/llm/prompts.py`.
Each prompt includes a `PROMPT_VERSION:` header; the pipeline logs which
version was used for each extraction. Use `dry_run: true` to preview.

---

## License

See [LICENSE](LICENSE).

---

## Non-goals for v1.x

- Team / multi-user support
- Mobile / browser extensions
- Cloud backend, sync, or hosted search
- Live integrations with ChatGPT or Claude services
- Autonomous agent actions that modify the corpus without explicit user request

This is a personal local tool. If it ever stops feeling that way, file an
issue.
