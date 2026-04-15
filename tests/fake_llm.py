"""Fake local LLM for tests.

Reads the prompt from stdin and emits a canned JSON response depending on
what the prompt is for. Good enough to exercise the subprocess path and
response parsing without needing a real model.

Usage pattern (tests configure local_llm.json with this as the command):

    ["python3", "-m", "tests.fake_llm", "<mode>"]

Supported modes:
  - summary_ok        : returns {"summary": "Short topical summary."}
  - summary_malformed : returns a non-JSON string
  - group_ok          : returns {"name": "chs staffing planning"}
  - group_generic     : returns {"name": "various topics"}   # should be rejected
  - gate_split_true   : returns {"split": true, "reason": "different subject"}
  - gate_split_false  : returns {"split": false, "reason": "same topic"}
  - nonzero           : exit code 1 (simulates runtime error)
  - hang              : sleep 10s (used with short timeout to test timeout path)
  - echo              : write stdin back to stdout (dry-run-like)
"""

from __future__ import annotations

import json
import sys
import time


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "echo"
    # Consume stdin so subprocess doesn't block.
    _ = sys.stdin.read()

    if mode == "summary_ok":
        sys.stdout.write(json.dumps({"summary": "Short topical summary."}))
        return 0
    if mode == "summary_malformed":
        sys.stdout.write("not json at all")
        return 0
    if mode == "group_ok":
        sys.stdout.write(json.dumps({"name": "chs staffing planning"}))
        return 0
    if mode == "group_generic":
        sys.stdout.write(json.dumps({"name": "various topics"}))
        return 0
    if mode == "gate_split_true":
        sys.stdout.write(json.dumps({"split": True, "reason": "different subject"}))
        return 0
    if mode == "gate_split_false":
        sys.stdout.write(json.dumps({"split": False, "reason": "same topic"}))
        return 0
    if mode == "nonzero":
        sys.stderr.write("simulated failure")
        return 1
    if mode == "hang":
        time.sleep(10)
        return 0
    # default: echo
    sys.stdout.write("echo mode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
