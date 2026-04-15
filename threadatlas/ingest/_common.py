"""Shared parser utilities: input file resolution and timestamp parsing."""

from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def parse_timestamp(value) -> float | None:
    """Coerce common timestamp encodings into POSIX seconds.

    Accepts: float/int (seconds since epoch), ISO 8601 strings (with or without
    trailing 'Z'), or None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # ChatGPT emits seconds; if a value is suspiciously large assume ms.
        v = float(value)
        if v > 1e12:  # millisecond range
            return v / 1000.0
        return v
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Strip nanoseconds beyond microsecond precision.
        s = re.sub(r"(\.\d{6})\d+", r"\1", s)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s).astimezone(timezone.utc).timestamp()
        except ValueError:
            return None
    return None


def _decode_json_bytes(data: bytes, source_label: str):
    """Decode JSON bytes tolerantly.

    Handles UTF-8 with or without BOM. If decoding fails, raise a
    ``ValueError`` that includes the source label so operators know which
    archive is malformed.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"Could not UTF-8 decode {source_label}: {e}") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in {source_label}: {e}") from e


def read_json_input(path: Path, expected_basename: str) -> tuple[list | dict, str]:
    """Resolve a path to JSON content for the conversations file.

    ``path`` may be:
    * a path to ``expected_basename`` itself
    * a directory containing ``expected_basename``
    * a .zip archive containing ``expected_basename`` somewhere inside

    Returns ``(payload, label)`` where ``label`` is a stable provenance string
    that includes the source archive basename if any.
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            for name in zf.namelist():
                if Path(name).name == expected_basename:
                    with zf.open(name) as fh:
                        data = fh.read()
                    label = f"{p.name}!{name}"
                    return _decode_json_bytes(data, label), label
        raise FileNotFoundError(
            f"{expected_basename} not found inside zip {p}"
        )
    if p.is_dir():
        cand = p / expected_basename
        if cand.exists():
            return _decode_json_bytes(cand.read_bytes(), str(cand)), str(cand)
        # search recursively (some exports nest in a subfolder)
        for found in p.rglob(expected_basename):
            return _decode_json_bytes(found.read_bytes(), str(found)), str(found)
        raise FileNotFoundError(f"{expected_basename} not found under {p}")
    if p.is_file() and p.name == expected_basename:
        return _decode_json_bytes(p.read_bytes(), str(p)), str(p)
    if p.is_file() and p.suffix.lower() == ".json":
        # User pointed at the file but it has a different name; accept it.
        return _decode_json_bytes(p.read_bytes(), str(p)), str(p)
    raise FileNotFoundError(
        f"Could not locate {expected_basename} at {p} (file/dir/zip)."
    )
