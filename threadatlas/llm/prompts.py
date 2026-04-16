"""Prompt templates.

All prompts are defined here so they can be audited in one place and
regression-tested. They are designed for small local instruct models
(Qwen2.5-3B / Llama-3.2-3B / Phi-3.5-Mini class) at temperature ~0.1.

Each prompt demands strict JSON output and includes a negative exemplar
when noise is the common failure mode.
"""

from __future__ import annotations


# We use explicit ``<<TOKEN>>`` placeholders (not ``str.format``) so that
# literal JSON braces in the prompt body don't need escaping. Template
# substitution is an unambiguous ``.replace``.


def _render(template: str, **vars: str) -> str:
    out = template
    for k, v in vars.items():
        out = out.replace(f"<<{k}>>", v)
    return out


# --- summarize -------------------------------------------------------------

SUMMARIZE_PROMPT = """You are an assistant that writes short, concrete
summaries of AI chat conversations for an offline personal index.

Return exactly one JSON object and nothing else:
{"summary": "<2-3 sentences, past tense, topical, concrete>"}

Rules:
- Describe what the user was working on or asked about, not what the AI said.
- Keep it concrete: mention names of projects, people, or decisions when present.
- Do not begin with "This conversation" or "The user". Start with the topic.
- Do not speculate, add context not present, or praise the AI.
- If the conversation is trivial (a greeting, a single sentence), say so briefly.

Conversation title: <<TITLE>>

Messages (ordered, user: / assistant: prefixes):
<<MESSAGES>>

Your JSON:"""


def render_summarize_prompt(title: str, rendered_messages: str) -> str:
    return _render(SUMMARIZE_PROMPT,
                   TITLE=title or "(no title)",
                   MESSAGES=rendered_messages)


# --- group naming ----------------------------------------------------------

GROUP_NAME_PROMPT = """You are an assistant that names thematic clusters of
AI chat conversations for an offline personal index.

You will be given <<N>> summaries that a clustering algorithm grouped
together. Produce a 3-7 word noun phrase that names this theme.

Return exactly one JSON object and nothing else:
{"name": "<3-7 word noun phrase>"}

Rules:
- Noun phrase, not a sentence. No trailing period.
- Specific, not generic. Use names of projects, tools, or domains if they
  recur across the summaries.
- Do NOT use filler names like "various topics", "general discussion",
  "miscellaneous", "assorted threads".
- Lowercase except for proper nouns and known acronyms.

Summaries in this cluster:
<<SUMMARIES>>

Your JSON:"""


def render_group_name_prompt(summaries: list[str]) -> str:
    bulleted = "\n".join(f"- {s.strip()}" for s in summaries if s.strip())
    return _render(GROUP_NAME_PROMPT, N=str(len(summaries)), SUMMARIES=bulleted)


# --- chunk boundary gate ---------------------------------------------------
#
# Precision-first. The deterministic chunker proposes boundaries; the LLM
# can only REMOVE them. When in doubt, return ``split: false`` (i.e. merge).

CHUNK_GATE_PROMPT = """You are deciding whether two adjacent segments of an
AI chat conversation actually represent different topics, or whether they
are the same conversation thread continuing.

You will see the last few messages of segment A and the first few messages
of segment B.

Rules:
- Be strict. Return {"split": true} ONLY when B starts a clearly
  different topic from A. When in doubt, return {"split": false}.
- A follow-up question about the same subject is NOT a topic shift.
- Refining a previous answer is NOT a topic shift.
- "Actually, different question:" or a completely unrelated subject IS a
  topic shift.

Return exactly one JSON object and nothing else:
{"split": true | false, "reason": "<short justification, one line>"}

Segment A (tail):
<<TAIL>>

Segment B (head):
<<HEAD>>

Your JSON:"""


def render_chunk_gate_prompt(tail_messages: str, head_messages: str) -> str:
    return _render(CHUNK_GATE_PROMPT,
                   TAIL=tail_messages.strip(),
                   HEAD=head_messages.strip())


# --- message rendering helpers --------------------------------------------

def _get_field(m, name: str, default: str = "") -> str:
    """Get a field from a Message object, dict, or sqlite3.Row.

    Avoids the ``getattr(...) or m.get(...)`` pattern which breaks when
    the attribute exists but is falsy (e.g. ``content_text=""``).
    """
    val = getattr(m, name, None)
    if val is not None:
        return val
    # Fall back to dict-style access (sqlite3.Row, plain dict).
    if hasattr(m, "get"):
        return m.get(name, default)
    try:
        return m[name]
    except (KeyError, TypeError, IndexError):
        return default


def render_messages(
    messages: list,
    *,
    max_chars_per_message: int = 800,
    roles_to_include: tuple[str, ...] = ("user", "assistant"),
) -> str:
    """Turn a list of Message-like records into a role-prefixed block.

    Accepts ``Message`` dataclass instances, dicts, or ``sqlite3.Row``
    objects.

    * Each message prefixed by ``user:`` or ``assistant:``.
    * Individual messages truncated to ``max_chars_per_message``.
    * Roles outside ``roles_to_include`` are dropped.
    """
    lines: list[str] = []
    for m in messages:
        role = _get_field(m, "role", "")
        if role not in roles_to_include:
            continue
        text = _get_field(m, "content_text", "") or ""
        text = text.strip().replace("\n", " ")
        if len(text) > max_chars_per_message:
            text = text[: max_chars_per_message - 3].rstrip() + "..."
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)
