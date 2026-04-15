"""Parser plugin interface.

Parsers turn raw export bytes/files into a stream of normalized
:class:`ParsedConversation` objects.

The interface is intentionally narrow:

* ``can_handle(path)``: cheap inspection to decide if this parser can read a
  file or directory. Used by ``--source auto`` and to validate explicit
  ``--source X`` selections.
* ``iter_conversations(path)``: yield parsed conversations one at a time.

Adding a new source = drop a new module that subclasses ``Parser`` and call
``registry.register(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class ParsedMessage:
    ordinal: int
    role: str  # one of Role values
    content_text: str
    timestamp: float | None = None
    source_message_id: str | None = None
    content_structured: dict | None = None


@dataclass
class ParsedConversation:
    source: str
    source_conversation_id: str | None
    title: str
    created_at: float | None
    updated_at: float | None
    messages: list[ParsedMessage] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)


class Parser:
    name: str = ""

    def can_handle(self, path: Path) -> bool:
        raise NotImplementedError

    def iter_conversations(self, path: Path) -> Iterator[ParsedConversation]:
        raise NotImplementedError


class _Registry:
    def __init__(self) -> None:
        self._by_name: dict[str, Parser] = {}

    def register(self, parser: Parser) -> None:
        self._by_name[parser.name] = parser

    def get(self, name: str) -> Parser:
        if name not in self._by_name:
            raise KeyError(f"Unknown source parser: {name!r}. Known: {list(self._by_name)}")
        return self._by_name[name]

    def get_autodetect(self) -> Parser:
        return _AutodetectParser(list(self._by_name.values()))

    def all(self) -> Iterable[Parser]:
        return self._by_name.values()


registry = _Registry()


class _AutodetectParser(Parser):
    """Tries each registered parser's ``can_handle`` and dispatches to the first match."""

    name = "auto"

    def __init__(self, candidates: list[Parser]):
        self._candidates = candidates

    def can_handle(self, path: Path) -> bool:
        return any(p.can_handle(path) for p in self._candidates)

    def iter_conversations(self, path: Path) -> Iterator[ParsedConversation]:
        for p in self._candidates:
            if p.can_handle(path):
                yield from p.iter_conversations(path)
                return
        raise ValueError(f"No registered parser can handle: {path}")
