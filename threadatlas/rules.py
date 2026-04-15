"""Auto-classification rules.

A vault can opt into automatic down-classification on import by dropping
a ``<vault>/auto_rules.json`` file. The rules engine scans each imported
conversation and, if any rule matches, sets the initial state to
``private`` or ``quarantined`` instead of ``pending_review``. A matched
rule overrides ``--auto-approve``.

Design invariants
-----------------
* Rules only ever DOWN-classify. A rule can take a conversation from
  ``pending_review`` or ``indexed`` to ``private`` or ``quarantined``,
  never the other way round. No rule can make something MORE visible.
* The outcome is deterministic: same rules + same content -> same state.
* Every match is recorded in the conversation's ``notes_local`` so the
  operator can see why each thread landed where it did.
* Rules operate on title / summary / message bodies, not on raw export
  metadata that might not be recoverable.

Config shape
------------

.. code-block:: json

    {
      "auto_private": [
        {"patterns": ["therapy", "anxiety medication"],
         "fields": ["title", "messages"]},
        {"patterns": ["\\\\b\\\\d{3}-\\\\d{2}-\\\\d{4}\\\\b"],
         "mode": "regex", "fields": ["messages"]}
      ],
      "auto_quarantine": [
        {"patterns": ["[no-index]"], "fields": ["title"]}
      ]
    }

Each rule:
  - ``patterns``: list of strings. Interpreted as case-insensitive
    substring matches by default.
  - ``mode``: ``"keyword"`` (default, substring) or ``"regex"``.
  - ``case_sensitive``: bool, default False.
  - ``fields``: list; any of ``title``, ``summary``, ``messages``.
    Default is all three.

If the config file is missing, rules are disabled and imports behave
normally. If it exists but has any validation errors, :func:`load_rules`
raises with a message that names the bad rule - fail loud, don't
silently let sensitive content leak into ``indexed``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .core.models import State


AUTO_RULES_BASENAME = "auto_rules.json"

_ALLOWED_FIELDS = frozenset({"title", "summary", "messages"})
# Severity ordering: which state "wins" if multiple rules match. More
# restrictive wins.
_SEVERITY = {
    State.PENDING_REVIEW.value: 0,
    State.INDEXED.value: 0,
    State.PRIVATE.value: 1,
    State.QUARANTINED.value: 2,
}


@dataclass(frozen=True)
class Rule:
    target_state: str  # "private" or "quarantined"
    patterns: tuple[re.Pattern, ...]
    fields: frozenset[str]
    raw_patterns: tuple[str, ...]  # retained for audit output
    mode: str  # "keyword" or "regex"


@dataclass
class RuleMatch:
    rule_target_state: str
    mode: str
    pattern: str
    field: str


@dataclass
class RuleSet:
    rules: tuple[Rule, ...] = field(default_factory=tuple)

    @property
    def empty(self) -> bool:
        return not self.rules


def rules_path(vault_root: Path) -> Path:
    return Path(vault_root) / AUTO_RULES_BASENAME


def load_rules(vault_root: Path) -> RuleSet:
    """Load rules from ``<vault>/auto_rules.json``; empty ``RuleSet`` if absent."""
    p = rules_path(vault_root)
    if not p.exists():
        return RuleSet()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"auto_rules.json: invalid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError("auto_rules.json: top-level must be an object")

    rules: list[Rule] = []
    for target in ("auto_private", "auto_quarantine"):
        entries = raw.get(target, []) or []
        if not isinstance(entries, list):
            raise ValueError(f"auto_rules.json: {target!r} must be a list")
        target_state = (
            State.PRIVATE.value if target == "auto_private"
            else State.QUARANTINED.value
        )
        for i, entry in enumerate(entries):
            rules.append(_compile_rule(entry, target_state, f"{target}[{i}]"))
    return RuleSet(rules=tuple(rules))


def _compile_rule(entry: dict, target_state: str, ctx: str) -> Rule:
    if not isinstance(entry, dict):
        raise ValueError(f"auto_rules.json: {ctx} must be an object")
    patterns_raw = entry.get("patterns") or []
    if (not isinstance(patterns_raw, list)
            or not patterns_raw
            or not all(isinstance(p, str) for p in patterns_raw)):
        raise ValueError(
            f"auto_rules.json: {ctx}.patterns must be a non-empty list of strings"
        )
    mode = entry.get("mode", "keyword")
    if mode not in ("keyword", "regex"):
        raise ValueError(f"auto_rules.json: {ctx}.mode must be 'keyword' or 'regex'")
    case_sensitive = bool(entry.get("case_sensitive", False))
    fields_raw = entry.get("fields") or ["title", "summary", "messages"]
    if not isinstance(fields_raw, list) or not fields_raw:
        raise ValueError(f"auto_rules.json: {ctx}.fields must be a non-empty list")
    for f in fields_raw:
        if f not in _ALLOWED_FIELDS:
            raise ValueError(
                f"auto_rules.json: {ctx}.fields contains unknown field {f!r}; "
                f"allowed: {sorted(_ALLOWED_FIELDS)}"
            )
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled: list[re.Pattern] = []
    for p in patterns_raw:
        try:
            if mode == "keyword":
                compiled.append(re.compile(re.escape(p), flags))
            else:
                compiled.append(re.compile(p, flags))
        except re.error as e:
            raise ValueError(
                f"auto_rules.json: {ctx}.patterns[{p!r}] is not a valid pattern: {e}"
            ) from e
    return Rule(
        target_state=target_state,
        patterns=tuple(compiled),
        fields=frozenset(fields_raw),
        raw_patterns=tuple(patterns_raw),
        mode=mode,
    )


def _text_for_field(
    field: str, *, title: str, summary: str, messages: Iterable[str]
) -> Iterable[str]:
    if field == "title":
        yield title or ""
    elif field == "summary":
        yield summary or ""
    elif field == "messages":
        for m in messages:
            if m:
                yield m


def evaluate(
    ruleset: RuleSet,
    *,
    title: str,
    summary: str = "",
    messages: Iterable[str] = (),
) -> tuple[str | None, list[RuleMatch]]:
    """Evaluate rules against conversation text.

    Returns ``(target_state, matches)``. ``target_state`` is the most
    restrictive matched state (or None if nothing matched). ``matches``
    lists every rule match for audit.
    """
    if ruleset.empty:
        return None, []
    # Materialize messages once; some callers may pass a generator.
    msg_list = [m or "" for m in messages]
    matches: list[RuleMatch] = []
    best_state: str | None = None
    for rule in ruleset.rules:
        for field in rule.fields:
            for text in _text_for_field(
                field, title=title, summary=summary, messages=msg_list,
            ):
                for pat, raw in zip(rule.patterns, rule.raw_patterns):
                    if pat.search(text):
                        matches.append(RuleMatch(
                            rule_target_state=rule.target_state,
                            mode=rule.mode,
                            pattern=raw,
                            field=field,
                        ))
                        if (best_state is None
                                or _SEVERITY[rule.target_state] > _SEVERITY[best_state]):
                            best_state = rule.target_state
                        break  # one match per rule/field is enough
    return best_state, matches


def summarize_matches(matches: list[RuleMatch]) -> str:
    """Render a short audit string suitable for ``notes_local``."""
    if not matches:
        return ""
    parts = [
        f"[{m.rule_target_state}] {m.mode}:{m.pattern!r} in {m.field}"
        for m in matches[:5]
    ]
    more = f" (+{len(matches)-5} more)" if len(matches) > 5 else ""
    return "auto-rule match: " + "; ".join(parts) + more
