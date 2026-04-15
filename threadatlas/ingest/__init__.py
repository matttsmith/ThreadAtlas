"""Ingest layer.

Each supported source provides a :class:`Parser` implementation. The
:func:`get_parser` registry maps a source name (``"chatgpt"``, ``"claude"``,
or ``"auto"``) to a parser. Adding a new source means adding a new module and
registering it in :data:`PARSERS` - no other layer needs to change.
"""

from .base import Parser, ParsedConversation, ParsedMessage, registry
from . import chatgpt as _chatgpt  # noqa: F401  (registers parser)
from . import claude as _claude  # noqa: F401  (registers parser)
from .pipeline import import_path, ImportResult


def get_parser(source: str) -> Parser:
    """Return the parser for a source. Use ``"auto"`` to autodetect."""
    if source == "auto":
        return registry.get_autodetect()
    return registry.get(source)


__all__ = [
    "Parser",
    "ParsedConversation",
    "ParsedMessage",
    "get_parser",
    "import_path",
    "ImportResult",
]
