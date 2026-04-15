"""A fake LLM that reads its prompt and emits plausible outputs.

Unlike ``tests/fake_llm.py`` (which always returns canned strings), this
module implements a very small "topical" fake that:

* For the summarize prompt, extracts the most frequent non-stopword
  tokens from the user messages and produces a summary that mentions
  them. This lets us test that our prompt tokenization / truncation /
  message rendering actually flow through to the summary output.
* For the group-naming prompt, picks the top distinctive tokens across
  the cluster's summaries and returns them as a short noun phrase.
* For the chunk boundary gate, compares the TAIL and HEAD message
  blocks: if their vocabularies share >= ``SAME_TOPIC_JACCARD``, it
  says ``split: false`` ("same topic"); otherwise ``split: true``.

This is NOT a model. It is deterministic, fast, and lets integration
tests verify that:

1. Prompts are rendered with the right content for the task.
2. The pipeline correctly threads model output back into the DB.
3. The k-means clustering then finds sensible groups.
4. The LLM-gated chunker makes reasonable boundary decisions.

Usage (from tests):

    [sys.executable, "-m", "tests.smart_fake_llm", "<mode>"]

Modes:
  - summary        emit {"summary": "..."}  based on user content
  - group_name     emit {"name":    "..."}  based on bulleted summaries
  - chunk_gate     emit {"split": bool, "reason": "..."}
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter


_TOKEN_RX = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "have",
    "has", "your", "you", "are", "was", "were", "but", "not", "any",
    "all", "can", "could", "would", "should", "what", "when", "where",
    "which", "while", "about", "they", "them", "their", "there", "here",
    "then", "than", "also", "just", "like", "want", "need", "make",
    "made", "use", "using", "used", "one", "two", "three", "lot", "more",
    "most", "some", "such", "very", "well", "yes", "okay", "thanks",
    "please", "really", "actually", "maybe", "got", "will", "shall",
    "may", "might", "does", "did", "doing", "done", "been", "being",
    "its", "its", "let", "lets", "going", "sure", "sorry", "try",
    "still", "now", "next", "out", "back", "down", "over", "same",
    "another", "both", "each", "every", "other", "how", "why",
    "user", "assistant", "system", "message", "messages",
    "gonna", "tell", "told", "say", "said", "ask", "asked", "get",
    "getting", "give", "given", "take", "taken", "taking",
    "think", "thought", "know", "knew", "seems", "seem", "looks",
    "looking", "help", "helping", "work", "working", "kind",
    "different", "various", "topic", "topics", "general", "stuff",
}


# ---------------------------------------------------------------------------
# Content analysis
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RX.findall(text or "")]


def _user_content(prompt: str) -> str:
    """Extract only the user-tagged content from a rendered conversation prompt.

    Our prompts use ``user:`` / ``assistant:`` prefixes; we take the user
    lines so the output focuses on what the human was working on (mirrors
    the real prompt's instruction).
    """
    lines = prompt.splitlines()
    out = []
    for ln in lines:
        if ln.strip().lower().startswith("user:"):
            out.append(ln.split(":", 1)[1])
    return "\n".join(out) if out else prompt


def _top_tokens(text: str, n: int = 6) -> list[str]:
    toks = [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) >= 3]
    counts = Counter(toks)
    return [t for t, _ in counts.most_common(n)]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

def _summary_from_prompt(prompt: str) -> str:
    user_text = _user_content(prompt)
    tops = _top_tokens(user_text, n=5)
    if not tops:
        return "Brief exchange with no clear topic."
    if len(tops) == 1:
        return f"Discussion focused on {tops[0]}."
    return (
        f"Discussion focused on {tops[0]} and {tops[1]}"
        + (", touching on " + ", ".join(tops[2:4]) if len(tops) >= 4 else "")
        + "."
    )


# ---------------------------------------------------------------------------
# Group naming
# ---------------------------------------------------------------------------

def _group_name_from_prompt(prompt: str) -> str:
    # The prompt passes summaries as bullet lines starting with "- ".
    bullets = [ln[2:] for ln in prompt.splitlines() if ln.startswith("- ")]
    blob = " ".join(bullets) if bullets else prompt
    tops = _top_tokens(blob, n=3)
    if not tops:
        return "small miscellaneous cluster"
    return " ".join(tops)


# ---------------------------------------------------------------------------
# Chunk gate
# ---------------------------------------------------------------------------

SAME_TOPIC_JACCARD = 0.18


def _chunk_gate_from_prompt(prompt: str) -> tuple[bool, str]:
    # Segments are delimited by the literal "Segment A (tail):" and
    # "Segment B (head):" markers from the prompt template.
    parts = prompt.split("Segment B (head):", 1)
    if len(parts) != 2:
        return False, "could not parse segments"
    tail_block = parts[0].split("Segment A (tail):", 1)
    if len(tail_block) != 2:
        return False, "could not parse tail"
    tail = tail_block[1]
    head = parts[1]
    tail_toks = set(t for t in _tokenize(tail) if t not in _STOPWORDS)
    head_toks = set(t for t in _tokenize(head) if t not in _STOPWORDS)
    j = _jaccard(tail_toks, head_toks)
    if j >= SAME_TOPIC_JACCARD:
        return False, f"same topic (jaccard={j:.2f})"
    return True, f"topic shift (jaccard={j:.2f})"


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    prompt = sys.stdin.read()
    if mode == "summary":
        sys.stdout.write(json.dumps({"summary": _summary_from_prompt(prompt)}))
    elif mode == "group_name":
        sys.stdout.write(json.dumps({"name": _group_name_from_prompt(prompt)}))
    elif mode == "chunk_gate":
        split, reason = _chunk_gate_from_prompt(prompt)
        sys.stdout.write(json.dumps({"split": split, "reason": reason}))
    else:
        sys.stderr.write(f"unknown mode: {mode}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
