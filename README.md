# ThreadAtlas

**Let Claude search your ChatGPT and Claude conversation history.** Fully local, no cloud APIs.

![tests](https://github.com/matttsmith/threadatlas/actions/workflows/test.yml/badge.svg)

---

## What it does

You export your chats from ChatGPT or Claude. ThreadAtlas indexes them
locally and exposes them to Claude Desktop via MCP. Claude can then
search your past conversations, see your active projects, decisions,
open questions, and generate a profile of what you've been working on.

Everything stays on your machine. No data leaves your computer.

---

## Setup (5 minutes)

### 1. Install

```bash
git clone https://github.com/matttsmith/threadatlas
cd threadatlas
pip install -e .
```

Requires Python 3.10+. Only dependency is `openpyxl` (spreadsheet export).

### 2. Create a vault and import your chats

```bash
# Create a vault (where ThreadAtlas stores its data).
threadatlas init ~/threadatlas-vault

# Import your ChatGPT export (zip file from Settings > Data controls > Export).
threadatlas import ~/threadatlas-vault ~/Downloads/chatgpt-export.zip --source chatgpt --auto-approve

# Import your Claude export (zip file from claude.ai Settings > Export).
threadatlas import ~/threadatlas-vault ~/Downloads/claude-export.zip --source claude --auto-approve
```

`--auto-approve` makes conversations immediately visible to Claude. Without
it, they land in `pending_review` and you approve them individually
(useful if some conversations are sensitive).

### 3. Process the conversations

```bash
# Build chunks + extract projects/decisions/entities (no LLM needed).
threadatlas process-approved ~/threadatlas-vault
```

This runs the deterministic extraction pipeline. It's fast and works
without any LLM.

### 4. Connect to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "threadatlas": {
      "command": "threadatlas",
      "args": ["mcp", "/absolute/path/to/threadatlas-vault"]
    }
  }
}
```

Restart Claude Desktop. The ThreadAtlas tools appear. Try asking Claude:
*"What have I been working on? Use ThreadAtlas to find out."*

---

## Optional: better extraction with a local LLM

The default extraction uses regex patterns — they work but aren't great
at distinguishing roleplay from real decisions. If you have a local
model running (llama-server, llamafile, Ollama, etc.), ThreadAtlas can
use it for much higher-quality extraction.

### Set up local_llm.json

Create `~/threadatlas-vault/local_llm.json`:

```json
{
  "provider": "llama_server",
  "base_url": "http://127.0.0.1:8080",
  "model": "your-model-name",
  "temperature": 0.1,
  "max_tokens": 512,
  "timeout_seconds": 120,
  "max_prompt_chars": 12000,
  "max_response_chars": 4000,
  "used_for": ["extraction", "turn_classification", "profile"],
  "dry_run": false
}
```

### Run the LLM extraction pipeline

```bash
# Validate the LLM is reachable.
threadatlas llm-check ~/threadatlas-vault --probe

# Run v2 extraction: classifies conversations, extracts structured data,
# generates summaries, builds search embeddings. Cached — safe to re-run.
threadatlas llm-extract ~/threadatlas-vault
```

This classifies each conversation by type (work, research, roleplay,
creative writing, etc.) and only extracts projects/decisions/open loops
from real work conversations. Roleplay dialogue like "I will spread my
evil!" no longer shows up as a decision.

**Time estimate**: ~20-30 seconds per conversation on a 3B model.
Results are cached in `~/.threadatlas/llm_cache.json`, so rebuilding
the vault doesn't re-run identical queries.

---

## What Claude sees (MCP tools)

13 read-only tools. Claude can typically profile a user in 2-3 calls.

| Tool | What it does |
|---|---|
| **`generate_profile`** | Narrative summary of who the user is: active projects, interests, open questions. The single most useful tool. |
| **`query`** | Search everything. Hybrid keyword + semantic. Filter by date, register, source. |
| **`find_related`** | "Given what the user is doing now, what past conversations are relevant?" |
| `get_conversation_summary` | Summary + metadata for one conversation. |
| `get_conversation_messages` | Raw messages for one conversation. |
| `get_conversation_chunks` | Thematic chunks for one conversation. |
| `list_projects` | Active projects (auto-excludes roleplay/jailbreak). |
| `get_project` | One project with linked conversations, decisions, entities. |
| `list_decisions` | User commitments and choices. |
| `list_open_loops` | Unresolved tasks and questions. |
| `list_entities` | People, organizations, concepts, artifacts. |
| `list_groups` | Thematic conversation clusters. |
| `get_group` | One cluster with its member conversations. |

All `list_*` tools accept optional `after`, `before`, `register`, and
`source` filters.

---

## Privacy model

Every conversation has a state:

| State | Visible to Claude? | Searchable in CLI? |
|---|---|---|
| `pending_review` | No | No |
| `indexed` | **Yes** | Yes |
| `private` | No | Yes (with `--include-private`) |
| `quarantined` | No | No |

`--auto-approve` on import sets everything to `indexed`. Without it,
you review and approve individually:

```bash
threadatlas review ~/threadatlas-vault        # see what's pending
threadatlas approve ~/threadatlas-vault <id>  # make visible to Claude
threadatlas private ~/threadatlas-vault <id>  # keep for CLI only
threadatlas delete ~/threadatlas-vault <id>   # hard delete (irreversible)
```

### Auto-classification rules

To auto-route sensitive conversations on import, create
`~/threadatlas-vault/auto_rules.json`:

```json
{
  "auto_private": [
    {"patterns": ["therapy", "therapist"], "fields": ["title", "messages"]}
  ],
  "auto_quarantine": [
    {"patterns": ["[secret]"], "fields": ["title"]}
  ]
}
```

Rules only down-classify. They always win over `--auto-approve`.

---

## CLI reference

```bash
# Import & review
threadatlas import <vault> <export> [--source chatgpt|claude] [--auto-approve]
threadatlas review <vault>
threadatlas approve / private / quarantine / delete <vault> <id>

# Process
threadatlas process-approved <vault>    # chunk + heuristic extract
threadatlas llm-extract <vault>         # v2 LLM pipeline (optional)
threadatlas group <vault>               # thematic clustering

# Search
threadatlas search <vault> "query"

# Serve
threadatlas mcp <vault>                 # stdio MCP server

# Maintenance
threadatlas check <vault>               # health check
threadatlas rebuild-from-normalized <vault>  # disaster recovery
```

Full reference: `threadatlas --help` and `threadatlas <cmd> --help`.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

424 tests covering parsing, state transitions, visibility, extraction,
search, MCP tools, LLM pipeline, and a static no-network guard.

---

## License

See [LICENSE](LICENSE).
