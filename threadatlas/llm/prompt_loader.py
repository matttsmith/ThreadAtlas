"""Load versioned prompts from the prompts/ directory.

Each prompt file has a ``PROMPT_VERSION: <version>`` header line.
Templates use ``<<TOKEN>>`` placeholders for substitution.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_VERSION_RX = re.compile(r"^PROMPT_VERSION:\s*(\S+)", re.MULTILINE)


def _load_raw(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def get_prompt_version(filename: str) -> str:
    raw = _load_raw(filename)
    m = _VERSION_RX.search(raw)
    return m.group(1) if m else "unknown"


def render_prompt(filename: str, **vars: str) -> str:
    raw = _load_raw(filename)
    out = raw
    for k, v in vars.items():
        out = out.replace(f"<<{k}>>", v)
    return out


# Convenience constants for prompt filenames.
TURN_CLASSIFIER_PROMPT = "turn_classifier_v1.txt"
EXTRACTION_PROMPT = "extraction_v1.txt"
PROFILE_PROMPT = "profile_v1.txt"
COMBINED_PROMPT = "combined_v1.txt"
