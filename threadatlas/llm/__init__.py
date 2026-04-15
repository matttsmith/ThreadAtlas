"""Local LLM integration.

Subprocess-only, stdio-only. There is no HTTP, no network, and no
third-party SDK. A user-configured local command (llama.cpp, MLX-LM,
llamafile, etc.) is invoked via ``subprocess.run`` with the prompt passed
via stdin or argv substitution.

The LLM is used narrowly:

* :mod:`.summarize` - topical summaries of conversations
* :mod:`.label_groups` - prose names for clusters (optional)
* :mod:`.chunking` - *gate* on deterministic chunk boundaries (only merges,
  never introduces new splits)

LLM usage is always opt-in. The config file at ``vault/local_llm.json``
must exist and its ``used_for`` whitelist must name the specific task, or
the runner will refuse to call the subprocess.
"""

from .config import LLMConfig, load_config
from .runner import LLMRunner, LLMError, LLMResponse

__all__ = [
    "LLMConfig",
    "load_config",
    "LLMRunner",
    "LLMError",
    "LLMResponse",
]
