"""Extraction precision tests using realistic-shape conversations.

These tests guard against the top false-positive sources observed in real
AI chat prose:

* casual "I like X" remarks should NOT create preferences
* advice-form "you need to X" should NOT create open loops
* "as a follow up question" should NOT create an open loop
* greetings ("Hi Claude", "Thanks John") should NOT become entities
* multi-match overlap on the same sentence should not spawn duplicates
"""

from __future__ import annotations

from threadatlas.core.models import DerivedKind, State
from threadatlas.core.workflow import transition_state
from threadatlas.extract import chunk_conversation, extract_for_conversation
from threadatlas.ingest import import_path


def _import_indexed(tmp_vault, store, factory, messages):
    path = factory([{"title": "Quality thread", "messages": messages}])
    res = import_path(tmp_vault, store, path)
    cid = res.imported[0]
    transition_state(store, cid, State.INDEXED.value)
    chunk_conversation(store, cid)
    return cid


def _kinds(store, cid) -> dict[str, list[str]]:
    rows = store.conn.execute(
        """
        SELECT o.kind, o.title
          FROM derived_objects o
          JOIN provenance_links p ON p.object_id = o.object_id
         WHERE p.conversation_id = ?
        """,
        (cid,),
    ).fetchall()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["kind"], []).append(r["title"])
    return out


def test_casual_like_does_not_become_preference(tmp_vault, store, chatgpt_export_factory):
    cid = _import_indexed(tmp_vault, store, chatgpt_export_factory, [
        ("user", "I like that approach, thanks!", 1.0),
        ("assistant", "Glad it helps.", 2.0),
        ("user", "I like it when responses are concise.", 3.0),
        ("assistant", "OK.", 4.0),
    ])
    extract_for_conversation(store, cid)
    kinds = _kinds(store, cid)
    assert kinds.get(DerivedKind.PREFERENCE.value, []) == [], \
        f"'I like' should not create preferences: {kinds.get(DerivedKind.PREFERENCE.value)}"


def test_stable_preference_is_extracted(tmp_vault, store, chatgpt_export_factory):
    cid = _import_indexed(tmp_vault, store, chatgpt_export_factory, [
        ("user", "I prefer async communication over meetings.", 1.0),
        ("assistant", "Noted.", 2.0),
        ("user", "My rule is to avoid calls before noon.", 3.0),
        ("assistant", "OK.", 4.0),
    ])
    extract_for_conversation(store, cid)
    kinds = _kinds(store, cid)
    prefs = kinds.get(DerivedKind.PREFERENCE.value, [])
    assert len(prefs) >= 2
    assert any("prefer" in p.lower() for p in prefs)


def test_advice_form_you_need_to_is_not_open_loop(tmp_vault, store, chatgpt_export_factory):
    cid = _import_indexed(tmp_vault, store, chatgpt_export_factory, [
        ("user", "How do I ship this?", 1.0),
        ("assistant", "You need to run the migration first.", 2.0),
        ("user", "OK thanks.", 3.0),
        ("assistant", "You need to verify the backup.", 4.0),
    ])
    extract_for_conversation(store, cid)
    kinds = _kinds(store, cid)
    loops = kinds.get(DerivedKind.OPEN_LOOP.value, [])
    # The assistant's "you need to" is advice, not the user's open loop.
    assert not any("you need to" in l.lower() for l in loops), f"found: {loops}"


def test_first_person_need_to_is_open_loop(tmp_vault, store, chatgpt_export_factory):
    cid = _import_indexed(tmp_vault, store, chatgpt_export_factory, [
        ("user", "I need to email Alice before Friday.", 1.0),
        ("assistant", "OK.", 2.0),
        ("user", "TODO: schedule Q2 planning.", 3.0),
        ("assistant", "Noted.", 4.0),
    ])
    extract_for_conversation(store, cid)
    kinds = _kinds(store, cid)
    loops = kinds.get(DerivedKind.OPEN_LOOP.value, [])
    assert any("i need to" in l.lower() for l in loops), f"missing first-person loop: {loops}"
    assert any("todo" in l.lower() for l in loops), f"missing TODO: {loops}"


def test_followup_question_phrase_not_open_loop(tmp_vault, store, chatgpt_export_factory):
    cid = _import_indexed(tmp_vault, store, chatgpt_export_factory, [
        ("user", "Quick note, as a follow up question — what about edge cases?", 1.0),
        ("assistant", "Good question.", 2.0),
        ("user", "Got it.", 3.0),
        ("assistant", "Anything else?", 4.0),
    ])
    extract_for_conversation(store, cid)
    kinds = _kinds(store, cid)
    loops = kinds.get(DerivedKind.OPEN_LOOP.value, [])
    # "follow up question" shouldn't hit our open loop patterns (we only fire
    # on "follow up on/with/about X", not bare "follow up").
    assert not any("follow up question" in l.lower() for l in loops), f"found: {loops}"


def test_greeting_names_not_entities(tmp_vault, store, chatgpt_export_factory):
    cid = _import_indexed(tmp_vault, store, chatgpt_export_factory, [
        ("user", "Hi Claude, thanks for the help.", 1.0),
        ("assistant", "Hello there. Thanks John.", 2.0),
        ("user", "Hi Claude, one more thing.", 3.0),
        ("assistant", "Sure John, go ahead.", 4.0),
    ])
    extract_for_conversation(store, cid)
    kinds = _kinds(store, cid)
    ents = kinds.get(DerivedKind.ENTITY.value, [])
    for name in ents:
        first = name.split()[0]
        assert first not in {"Hi", "Hello", "Thanks", "Thank", "Sure", "Dear"}, \
            f"greeting leaked into entity: {name}"


def test_real_named_entity_still_extracted(tmp_vault, store, chatgpt_export_factory):
    cid = _import_indexed(tmp_vault, store, chatgpt_export_factory, [
        ("user", "I talked to Alice Johnson yesterday about the plan.", 1.0),
        ("assistant", "What did Alice Johnson say?", 2.0),
        ("user", "Alice Johnson agreed to review by Friday.", 3.0),
        ("assistant", "OK.", 4.0),
    ])
    extract_for_conversation(store, cid)
    kinds = _kinds(store, cid)
    ents = kinds.get(DerivedKind.ENTITY.value, [])
    assert any("alice johnson" in e.lower() for e in ents), f"missing Alice Johnson: {ents}"


def test_overlapping_decision_patterns_dedup(tmp_vault, store, chatgpt_export_factory):
    cid = _import_indexed(tmp_vault, store, chatgpt_export_factory, [
        ("user", "I decided to go with the smaller team to ship faster.", 1.0),
        ("assistant", "Good choice.", 2.0),
        ("user", "OK more discussion.", 3.0),
        ("assistant", "Noted.", 4.0),
    ])
    extract_for_conversation(store, cid)
    kinds = _kinds(store, cid)
    decisions = kinds.get(DerivedKind.DECISION.value, [])
    # Multiple decision patterns may match the same sentence; after dedup,
    # we expect at most one decision from it.
    distinct = {d.strip().lower()[:60] for d in decisions}
    assert len(distinct) == len(decisions), f"duplicate decisions persisted: {decisions}"
