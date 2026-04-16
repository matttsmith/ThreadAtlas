"""Local LLM integration.

Supports two backends:

* **subprocess** (default) — invokes a user-configured local command
  (llama.cpp CLI, MLX-LM, llamafile, etc.) via ``subprocess.run`` with the
  prompt passed via stdin or argv substitution.  No network, no HTTP.

* **llama_server** — sends requests to a locally-running llama-server (or
  any OpenAI-compatible endpoint on loopback) via ``/v1/chat/completions``.
  Rejects non-loopback URLs unless explicitly allowed.

The LLM is used narrowly:

* :mod:`.summarize` — topical summaries of conversations
* :mod:`.label_groups` — prose names for clusters (optional)
* :mod:`.chunking` — *gate* on deterministic chunk boundaries (only merges,
  never introduces new splits)

LLM usage is always opt-in.  The config file at ``vault/local_llm.json``
must exist and its ``used_for`` whitelist must name the specific task, or
the runner will refuse to invoke the backend.
"""

from .config import LLMConfig, load_config
from .common import LLMResponse, parse_json_response
from .errors import LLMError, LLMNotConfiguredError
from .runner import LLMRunner

__all__ = [
    "LLMConfig",
    "load_config",
    "LLMRunner",
    "LLMError",
    "LLMNotConfiguredError",
    "LLMResponse",
    "parse_json_response",
]
