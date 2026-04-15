# ThreadAtlas design spec

ThreadAtlas is a fully local conversation intelligence layer for ChatGPT and
Claude exports. This document is the operator-facing condensation of the
v1 design.

## Goals (in priority order)

1. Local-only security posture.
2. Review and visibility controls.
3. Search quality.
4. Project reconstruction.
5. Provenance.
6. Deletion correctness.
7. Spreadsheet usefulness.

Anything else is secondary.

## Layered architecture

```
core/      domain models, vault layout, state transitions, deletion
ingest/    parser plugins (chatgpt, claude) + import pipeline
store/     SQLite + FTS5 + normalized JSON IO
extract/   chunking + heuristic derived objects
search/    keyword search + project synthesis
export/    XLSX workbook profiles
mcp/       stdio JSON-RPC adapter (read-only, indexed-only)
cli/       argparse subcommands
```

## States

* `pending_review` - just imported, hidden from MCP and synthesis
* `indexed` - searchable, eligible for synthesis, MCP-visible
* `private` - locally searchable in CLI, hidden from MCP and global synthesis
* `quarantined` - normalized only; chunks/embeddings/provenance stripped
* `deleted` - hard delete physically removes records

The CLI is the only path to change state. There are no mutating MCP tools.

## Vault layout

```
vault/
  raw_imports/    original user-provided export files
  normalized/     canonical per-conversation JSON (sharded)
  db/             SQLite database
  exports/        generated XLSX
  cache/          ephemeral, safe to rebuild
  logs/           local operational logs
```

The normalized file is the source of truth for conversation text. The DB
indexes over it.

## Adding a new source

Every parser is a class implementing
``threadatlas.ingest.base.Parser`` with two methods:

```python
def can_handle(self, path: Path) -> bool: ...
def iter_conversations(self, path: Path) -> Iterator[ParsedConversation]: ...
```

Drop a new module into ``threadatlas/ingest/``, register it via
``registry.register(MyParser())``, and import it from
``threadatlas/ingest/__init__.py``. No other layer needs to change.

## Privacy guarantees

* No outbound network access. The shipped package contains no networking
  imports; ``tests/test_no_network.py`` enforces this.
* MCP runs over stdio only. There is no inbound TCP listener.
* Every MCP tool filters to ``MCP_VISIBLE_STATES`` (indexed only).
* Hard delete physically removes messages, chunks, FTS rows, provenance,
  orphan derived objects, and the normalized file. ``VACUUM`` is run after
  delete.

## Test layout

* `tests/test_vault.py` - vault layout invariants
* `tests/test_parsers.py` - ChatGPT/Claude parser fidelity
* `tests/test_import_pipeline.py` - import lands in pending_review, dedupes
* `tests/test_state_transitions.py` - allowed/disallowed transitions
* `tests/test_chunking.py` - thematic boundaries, message alignment
* `tests/test_extraction.py` - heuristic derived objects per kind
* `tests/test_search.py` - visibility filtering during search
* `tests/test_deletion.py` - hard-delete cascade and orphan cleanup
* `tests/test_mcp.py` - JSON-RPC handler + visibility enforcement
* `tests/test_export.py` - XLSX sheet shape per profile
* `tests/test_cli.py` - end-to-end CLI lifecycle
* `tests/test_no_network.py` - static no-network assertion
