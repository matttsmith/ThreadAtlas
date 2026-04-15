# ThreadAtlas

A fully local, deletion-aware conversation intelligence layer for ChatGPT and
Claude exports.

ThreadAtlas ingests AI conversation exports into a local vault, lets you review
and partition them by sensitivity, supports keyword search, splits long threads
into thematic chunks, extracts a small set of high-value derived objects
(projects, people, decisions, open loops, artifacts, preferences), and exposes
the approved subset to Claude via a local stdio MCP server. It also ships
spreadsheet-quality XLSX export for manual triage.

It is a single-user local utility. There is no cloud backend, no telemetry, and
no outbound network access at runtime.

## Install

```
pip install -e .
```

This installs the `threadatlas` CLI. The only runtime dependency is
`openpyxl` (XLSX writing). The rest of the implementation is standard library
only. Optional local model integration is not bundled.

## Quick start

```
# 1. Initialize a vault
threadatlas init ./vault

# 2. Import a ChatGPT export (zip or directory or conversations.json)
threadatlas import ./vault chatgpt-export.zip --source chatgpt

# 3. Import a Claude export
threadatlas import ./vault claude-export.zip --source claude

# 4. Review what landed in pending_review
threadatlas review ./vault

# 5. Approve, quarantine, mark private, or delete
threadatlas approve ./vault <conversation_id>
threadatlas quarantine ./vault <conversation_id>
threadatlas private ./vault <conversation_id>
threadatlas delete ./vault <conversation_id>

# 6. Build chunks and run heuristic extraction (no network, no LLM required)
threadatlas chunk ./vault
threadatlas extract ./vault

# 7. Search
threadatlas search ./vault "project chs decision"

# 8. Export to XLSX
threadatlas export ./vault --profile review_workbook --out ./vault/exports/review.xlsx

# 9. Inspect a single conversation end to end
threadatlas inspect ./vault <conversation_id>

# 10. Run the local MCP server (stdio)
threadatlas mcp ./vault
```

## Vault layout

```
vault/
  raw_imports/    # original user-provided export files
  normalized/     # canonical normalized conversation JSON
  db/             # SQLite database + indexes
  exports/        # generated XLSX
  cache/          # ephemeral, safe to rebuild
  logs/           # local operational logs
```

## Visibility states

- `pending_review` - just imported, not searchable by MCP, not in synthesis
- `indexed` - searchable, eligible for synthesis, MCP-visible
- `private` - locally searchable in the CLI but hidden from MCP and global synthesis
- `quarantined` - normalized only, no embeddings/extraction, hidden from MCP
- `deleted` - physically removed (CLI command performs hard delete)

## Privacy guarantees

ThreadAtlas does not import any HTTP or networking client in its runtime code
paths. A regression test (`tests/test_no_network.py`) statically scans the
package source for forbidden imports.

If you opt into local embeddings or a local LLM later, you must install and
configure them manually. ThreadAtlas will never download a model.

See `docs/spec.md` for the full design spec.
