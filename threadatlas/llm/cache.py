"""Persistent LLM response cache.

Stores a mapping of ``SHA-256(prompt) -> raw_response`` in a JSON file
at ``~/.threadatlas/llm_cache.json``.  This file lives *outside* the
vault so it survives vault deletes and rebuilds.

The cache is keyed on the full prompt text (after truncation).  If the
prompt template changes, the hash changes and the old entry is simply
never hit — stale entries waste a little disk but never produce wrong
results.

Design notes:
* The file is append-friendly: we read the whole thing into memory on
  first access (it's a dict, not a log), mutate, and write back.  For
  the expected scale (tens of thousands of entries, each a few hundred
  bytes of response) this is fine.
* We do NOT cache failed responses — only ``success=True`` results are
  stored.  This means a transient timeout doesn't poison the cache.
* Thread safety is not a concern: ThreadAtlas is single-process.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DIR = Path.home() / ".threadatlas"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "llm_cache.json"

# Maximum cache entries before we start evicting oldest.  At ~500 bytes
# per entry this caps the file at roughly 50 MB, which is generous.
MAX_CACHE_ENTRIES = 100_000


def _cache_path() -> Path:
    return DEFAULT_CACHE_FILE


def _load_cache(path: Path | None = None) -> dict[str, Any]:
    p = path or _cache_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_cache(cache: dict[str, Any], path: Path | None = None) -> None:
    p = path or _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Evict oldest entries if over limit.
    if len(cache) > MAX_CACHE_ENTRIES:
        # Sort by insertion order (Python 3.7+ dicts are ordered) and keep
        # the most recent entries.
        items = list(cache.items())
        cache = dict(items[-MAX_CACHE_ENTRIES:])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def prompt_hash(prompt: str) -> str:
    """SHA-256 hex digest of the prompt text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def cache_get(prompt: str, *, path: Path | None = None) -> str | None:
    """Look up a cached response for the given prompt.

    Returns the raw response string on hit, ``None`` on miss.
    """
    h = prompt_hash(prompt)
    cache = _load_cache(path)
    entry = cache.get(h)
    if entry is None:
        return None
    # Entry format: {"raw": "...", "task": "..."} or just a string
    # (for forward-compat we accept both).
    if isinstance(entry, dict):
        return entry.get("raw")
    if isinstance(entry, str):
        return entry
    return None


def cache_put(prompt: str, raw_response: str, *, task: str = "",
              path: Path | None = None) -> None:
    """Store a successful response in the cache."""
    h = prompt_hash(prompt)
    cache = _load_cache(path)
    cache[h] = {"raw": raw_response, "task": task}
    _save_cache(cache, path)


def cache_stats(path: Path | None = None) -> dict[str, int]:
    """Return basic cache statistics."""
    cache = _load_cache(path)
    return {"entries": len(cache)}
