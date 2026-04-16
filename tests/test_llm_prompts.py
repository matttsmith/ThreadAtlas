"""Regression tests for LLM prompt rendering.

The critical fix: render_messages must handle both Message dataclass
instances and dict/Row-like objects without crashing on falsy attribute
values.
"""

from __future__ import annotations

from threadatlas.core.models import Message
from threadatlas.llm.prompts import render_messages


def test_render_messages_with_message_objects():
    msgs = [
        Message(message_id="m1", conversation_id="c1", ordinal=0,
                role="user", content_text="hello world"),
        Message(message_id="m2", conversation_id="c1", ordinal=1,
                role="assistant", content_text="hi there"),
    ]
    out = render_messages(msgs)
    assert "user: hello world" in out
    assert "assistant: hi there" in out


def test_render_messages_with_dicts():
    msgs = [
        {"role": "user", "content_text": "hello"},
        {"role": "assistant", "content_text": "hi"},
    ]
    out = render_messages(msgs)
    assert "user: hello" in out
    assert "assistant: hi" in out


def test_render_messages_with_empty_content_text():
    """Regression: Message with content_text='' must not crash on .get()."""
    msgs = [
        Message(message_id="m1", conversation_id="c1", ordinal=0,
                role="user", content_text=""),
        Message(message_id="m2", conversation_id="c1", ordinal=1,
                role="assistant", content_text="real reply"),
    ]
    out = render_messages(msgs)
    # Empty content_text should be skipped, not crash.
    assert "user:" not in out
    assert "assistant: real reply" in out


def test_render_messages_filters_roles():
    msgs = [
        Message(message_id="m1", conversation_id="c1", ordinal=0,
                role="system", content_text="you are helpful"),
        Message(message_id="m2", conversation_id="c1", ordinal=1,
                role="user", content_text="question"),
    ]
    out = render_messages(msgs)
    assert "system" not in out
    assert "user: question" in out


def test_render_messages_truncates_long_messages():
    long_text = "x" * 2000
    msgs = [
        Message(message_id="m1", conversation_id="c1", ordinal=0,
                role="user", content_text=long_text),
    ]
    out = render_messages(msgs, max_chars_per_message=100)
    assert len(out) < 200
    assert out.endswith("...")
