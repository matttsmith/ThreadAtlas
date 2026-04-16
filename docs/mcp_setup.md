# Connecting ThreadAtlas to Claude Desktop

ThreadAtlas ships a **stdio JSON-RPC MCP server** that any MCP-compatible
host can launch. The most common host is **Claude Desktop** (the native
Mac/Windows app from Anthropic).

## Quickest path: desktop extension

ThreadAtlas ships as a Claude Desktop extension (`.mcpb` file). This is
the easiest install:

1. Build the extension (requires `mcpb` CLI):
   ```bash
   npm install -g @anthropic-ai/mcpb
   mcpb pack . threadatlas.mcpb
   ```
2. In Claude Desktop, go to **Settings > Extensions > Advanced settings**.
3. Click **Install Extension** and select `threadatlas.mcpb`.
4. When prompted, set the **Vault Directory** to your vault path.
5. Done — the ThreadAtlas tools appear in your next conversation.

The extension uses `uv` to auto-install Python and dependencies. No
manual PATH or venv management needed.

## Manual setup (alternative)

If you prefer manual configuration, or are using a different MCP host
(Claude Code, an IDE extension, another desktop client), the principles
are the same: you tell the host to launch `threadatlas mcp <vault>` as
a subprocess on stdio.

---

## 1. Install ThreadAtlas and pick a vault

```bash
# In the ThreadAtlas repo:
pip install -e ".[dev]"

# Create (or pick) the vault Claude Desktop should see:
threadatlas init ~/ThreadAtlas/vaults/personal
```

Approve at least one conversation so Claude has something to look at:

```bash
threadatlas import ~/ThreadAtlas/vaults/personal ~/Downloads/chatgpt-export.zip
threadatlas review ~/ThreadAtlas/vaults/personal
threadatlas approve ~/ThreadAtlas/vaults/personal <conv_id>
threadatlas process-approved ~/ThreadAtlas/vaults/personal
```

Only `indexed` conversations are exposed through MCP. `pending_review`,
`private`, and `quarantined` content is never surfaced.

---

## 2. Find the absolute path of the `threadatlas` CLI

Claude Desktop launches subprocesses with a minimal `PATH` and **won't
find** binaries via shell aliases, `asdf`, `pyenv` shims, or per-shell
activations. Use the absolute path.

```bash
which threadatlas
# example output on macOS with a venv:
# /Users/matt/.venvs/threadatlas/bin/threadatlas

# example on a Homebrew Python:
# /opt/homebrew/bin/threadatlas
```

Note this path — you'll put it in the config in step 3.

If you're using a virtualenv, point at **the venv's** `threadatlas`
binary, not the system `threadatlas`. If you used `pipx` it will be at
`~/.local/bin/threadatlas`.

---

## 3. Edit `claude_desktop_config.json`

The config file lives at:

| Platform | Path |
|---|---|
| macOS   | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux   | `~/.config/Claude/claude_desktop_config.json` |

If the file doesn't exist, create it. Add an entry under `mcpServers`:

```json
{
  "mcpServers": {
    "threadatlas": {
      "command": "/Users/matt/.venvs/threadatlas/bin/threadatlas",
      "args": ["mcp", "/Users/matt/ThreadAtlas/vaults/personal"]
    }
  }
}
```

That is the whole config. Notes:

* The `command` **must be an absolute path**. No `~`, no `$HOME`, no
  PATH lookup.
* The second argument to `args` is the absolute path to the vault you
  want Claude to see. If you have multiple vaults, add multiple
  `mcpServers` entries with distinct keys (e.g. `threadatlas_work`,
  `threadatlas_personal`).
* No `env` block is required. If you use one, it **replaces** the
  launcher's env — so set `PATH` explicitly if anything needs it.

### Windows caveats

- Use double backslashes in JSON: `"C:\\Users\\matt\\...\\threadatlas.exe"`
- The vault path also needs double backslashes.
- `pip install -e .` on Windows installs `threadatlas.exe` into
  `Scripts\`. Point `command` there.

---

## 4. Restart Claude Desktop

Claude Desktop reads `claude_desktop_config.json` only at launch. Quit
the app completely (Cmd-Q / right-click the tray icon → Quit) and
reopen it.

---

## 5. Verify the connection

In Claude Desktop, start a new conversation. Claude will show a tools
indicator in the message composer when MCP servers are connected — the
ThreadAtlas tools should appear.

Try a sanity-check question:

> "Use the ThreadAtlas MCP server to list the first 5 indexed
> conversations and summarise what they're about."

Claude will call `search_conversations` / `get_conversation_summary` and
respond with results drawn from your vault.

The tools Claude can call are, by default, **read-only and indexed-only**:

- `search_conversations`, `search_chunks`
- `get_conversation_summary`, `get_conversation_messages`, `get_conversation_chunks`
- `list_projects`, `get_project`, `get_project_timeline`
- `list_open_loops`, `list_decisions`, `list_entities`
- `list_groups`, `get_group`
- `inspect_conversation_storage` (metadata only; redacts title for
  non-visible conversations)

Claude **cannot** change visibility state, delete conversations,
merge/suppress derived objects, or alter message content. Those remain
CLI-only, with confirmation prompts.

### Optional: narrow write tools

If you want Claude to be able to correct obviously-wrong labels and
tags, drop a `<vault>/mcp_config.json`:

```json
{"allow_writes": true}
```

Restart Claude Desktop. Four additional tools become available:

- `set_group_label` — correct a cluster's prose label.
- `add_tag` / `remove_tag` — manual tags on `indexed` conversations.
- `rename_derived_object` — fix an auto-extracted project/entity name.

Guarantees:
- **State changes are never exposed.** Claude still cannot move a
  conversation between `pending_review` / `indexed` / `private` /
  `quarantined` / `deleted`.
- `rename_derived_object` refuses to rename an object whose provenance
  is entirely in non-indexed conversations (can't be used to surface or
  edit private-only material).
- Every successful write is logged to
  `<vault>/logs/mcp_mutations.jsonl` with timestamp + tool + args +
  outcome. **No message content is logged.** You can audit what Claude
  did.
- String inputs are length-capped before hitting the DB; prompt
  injection trying to stuff huge payloads into labels is truncated.

---

## 6. Troubleshooting

### "No ThreadAtlas tools in Claude Desktop"

1. Check the config file actually parses as JSON. A trailing comma or
   stray character breaks all MCP servers silently.
   ```bash
   python -m json.tool < ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```
2. Confirm the command path exists and is executable:
   ```bash
   /Users/matt/.venvs/threadatlas/bin/threadatlas --version
   ```
3. Make sure the vault path exists and was initialised:
   ```bash
   threadatlas check /Users/matt/ThreadAtlas/vaults/personal
   ```
4. Restart Claude Desktop fully.

### Checking Claude Desktop's MCP logs

Claude Desktop writes one log per MCP server:

| Platform | Path |
|---|---|
| macOS   | `~/Library/Logs/Claude/mcp-server-threadatlas.log` |
| Windows | `%LOCALAPPDATA%\Claude\logs\mcp-server-threadatlas.log` |

Tail that file while restarting Claude Desktop:

```bash
tail -f ~/Library/Logs/Claude/mcp-server-threadatlas.log
```

A healthy startup shows the `initialize` handshake; errors come through
here with clear messages.

### "command not found" / python errors

Almost always means the `command` path is wrong or points at a Python
install that doesn't have `threadatlas` installed. `pipx install -e .`
from the repo creates a self-contained, reliably-discoverable binary at
`~/.local/bin/threadatlas` — that's the simplest option if your
pyenv/venv setup is complex.

### Run the server manually to reproduce errors

The stdio protocol is plain newline-delimited JSON-RPC. You can drive
it by hand:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | threadatlas mcp /path/to/vault
```

You should see two JSON lines come back. Any stack trace on stderr is
what Claude Desktop would also be seeing.

---

## 7. Privacy reminders

- MCP only ever exposes `indexed` content. `private` / `quarantined` /
  `pending_review` are never returned.
- Group labels sent through MCP are **recomputed** from only indexed
  members, so a mostly-private group cannot leak its contents through
  its label string.
- Claude Desktop still sends what you ask it to reason about to
  Anthropic's cloud API — that is how Claude itself works. MCP is a
  local tool bridge, not a privacy wrapper around Claude. Decisions
  about what to approve as `indexed` determine what Claude may see; be
  deliberate about that.
- If you later approve a conversation, Claude Desktop will see it on
  the next query. If you quarantine or delete one, the next query will
  no longer see it.

---

## 8. Multiple vaults, multiple Claude instances

You can register multiple ThreadAtlas MCP servers side-by-side:

```json
{
  "mcpServers": {
    "threadatlas_personal": {
      "command": "/Users/matt/.venvs/threadatlas/bin/threadatlas",
      "args": ["mcp", "/Users/matt/vaults/personal"]
    },
    "threadatlas_work": {
      "command": "/Users/matt/.venvs/threadatlas/bin/threadatlas",
      "args": ["mcp", "/Users/matt/vaults/work"]
    }
  }
}
```

Each vault runs as an independent subprocess. Claude can draw from
whichever is relevant for a given question; all of them remain
read-only and indexed-only.

---

## 9. Non-Claude-Desktop hosts

Any MCP host that launches servers over stdio works the same way. The
command is always:

```
<absolute path to threadatlas>  mcp  <absolute path to vault>
```

Cursor, Continue, Zed, and other IDEs with MCP support all follow the
same config pattern (different file location, same `command` + `args`
schema). Check the host's docs for the config file location.
