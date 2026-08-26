"""Autonomy, provenance, and grounded-retrieval contracts for Holographic memory."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


def test_provenance_events_and_explicit_entities_round_trip(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", hrr_dim=64)
    try:
        fact_id = store.add_fact(
            "Falko prefers direct operational answers.",
            category="user_pref",
            entities=["Falko"],
            provenance={
                "source_type": "user_explicit",
                "session_id": "discord:test",
                "evidence": "I prefer direct operational answers.",
                "volatility": "stable",
            },
        )
        fact = store.get_fact(fact_id)
        assert fact["entities"] == ["Falko"]
        assert fact["provenance"][0]["source_type"] == "user_explicit"
        assert fact["provenance"][0]["session_id"] == "discord:test"

        event_id = store.enqueue_event(
            "retention_candidate",
            session_id="discord:test",
            payload={"content": "I always want failures stated plainly."},
        )
        pending = store.list_events(status="pending")
        assert [event["event_id"] for event in pending] == [event_id]
        assert json.loads(pending[0]["payload_json"])["content"].startswith("I always")

        resolved = store.resolve_event(event_id, "applied", fact_id=fact_id)
        assert resolved["status"] == "applied"
        assert store.list_events(status="pending") == []
    finally:
        store.close()


def test_remove_resolved_fact_detaches_event_and_commits_one_audit_record(tmp_path):
    store = MemoryStore(tmp_path / "remove-resolved.db", hrr_dim=64)
    try:
        fact_id = store.add_fact("Temporary fact", category="project")
        event_id = store.enqueue_event("retention_candidate", payload={"fact_id": fact_id})
        store.resolve_event(event_id, "applied", fact_id=fact_id)

        assert store.remove_fact(fact_id) is True
        assert store.get_fact(fact_id) is None
        event = store._conn.execute(
            "SELECT status, resolved_fact_id FROM memory_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        assert event["status"] == "applied"
        assert event["resolved_fact_id"] is None
        audit_count = store._conn.execute(
            "SELECT COUNT(*) FROM fact_events WHERE fact_id = ? AND event_type = 'fact_removed'",
            (fact_id,),
        ).fetchone()[0]
        assert audit_count == 1
    finally:
        store.close()


def test_probe_and_reason_are_grounded_in_explicit_entity_links(tmp_path):
    store = MemoryStore(tmp_path / "grounded.db", hrr_dim=64)
    try:
        both = store.add_fact(
            "Falko uses SmartSalary for salary packaging.",
            entities=["Falko", "SmartSalary"],
        )
        falko_only = store.add_fact(
            "Falko develops SpaceSim.",
            entities=["Falko", "SpaceSim"],
        )
        salary_only = store.add_fact(
            "SmartSalary provides packaging services.",
            entities=["SmartSalary"],
        )
        unrelated = store.add_fact(
            "LG makes televisions.",
            entities=["LG"],
        )
        tag_only_quarantine = store.add_fact(
            "Falko once had a superseded temporary preference.",
            entities=["Falko"],
            tags="dream-quarantine,duplicate-of:1",
        )
        retriever = FactRetriever(store, hrr_dim=64)

        assert tag_only_quarantine not in {
            row["fact_id"]
            for row in retriever.search("superseded temporary", category="general", limit=20)
        }
        assert tag_only_quarantine not in set(
            store.fact_ids_for_entity("Falko", category="general")
        )

        probe_ids = {row["fact_id"] for row in retriever.probe("Falko", limit=20)}
        assert probe_ids == {both, falko_only}
        reason_ids = {
            row["fact_id"]
            for row in retriever.reason(["Falko", "SmartSalary"], limit=20)
        }
        assert reason_ids == {both}
        assert salary_only not in probe_ids
        assert unrelated not in probe_ids
    finally:
        store.close()


def test_audit_retrieval_does_not_change_live_usage_telemetry(tmp_path):
    store = MemoryStore(tmp_path / "telemetry.db", hrr_dim=64)
    try:
        fact_id = store.add_fact("Falko prefers bounded triage.", entities=["Falko"])
        audit_retriever = FactRetriever(store, hrr_dim=64, track_usage=False)
        assert audit_retriever.search("bounded triage")
        assert audit_retriever.probe("Falko")
        row = store.get_fact(fact_id)
        assert row["retrieval_count"] == 0
        assert row["shown_count"] == 0
        assert row["last_retrieved_at"] is None

        live_retriever = FactRetriever(store, hrr_dim=64)
        assert live_retriever.search("bounded triage")
        row = store.get_fact(fact_id)
        assert row["retrieval_count"] == 1
        assert row["shown_count"] == 1
        assert row["last_retrieved_at"] is not None
    finally:
        store.close()


def test_reversible_quarantine_records_relation_and_protects_preferences(tmp_path):
    store = MemoryStore(tmp_path / "quarantine.db", hrr_dim=64)
    try:
        canonical = store.add_fact("The project uses pytest.", category="project")
        duplicate = store.add_fact("Project tests run with pytest.", category="project")
        result = store.quarantine_fact(
            duplicate,
            related_fact_id=canonical,
            relation_type="duplicate_of",
            evidence="Same durable test-runner claim; canonical fact is complete.",
        )
        assert result["fact"]["category"] == "quarantine"
        assert result["fact"]["original_category"] == "project"
        assert result["fact"]["original_trust_score"] == pytest.approx(0.5)
        assert result["relation"]["relation_type"] == "duplicate_of"
        assert f"duplicate-of:{canonical}" in result["fact"]["tags"]

        restored = store.restore_fact(
            duplicate,
            evidence="The quarantine decision was intentionally reverted for verification.",
        )
        assert restored["fact"]["category"] == "project"
        assert restored["fact"]["trust_score"] == pytest.approx(0.5)
        assert restored["fact"]["relations"][0]["active"] == 0
        event_types = [event["event_type"] for event in restored["fact"]["events"]]
        assert "fact_quarantined" in event_types
        assert "fact_restored" in event_types

        protected = store.add_fact(
            "Falko prefers direct answers.",
            category="user_pref",
            entities=["Falko"],
        )
        with pytest.raises(ValueError, match="protected"):
            store.quarantine_fact(
                protected,
                related_fact_id=canonical,
                relation_type="duplicate_of",
                evidence="Agent inference is not direct correction evidence.",
            )
    finally:
        store.close()


def test_sync_turn_queues_only_strict_direct_durable_candidates(tmp_path):
    provider = HolographicMemoryProvider(
        config={
            "db_path": str(tmp_path / "provider.db"),
            "hrr_dim": 64,
            "event_capture": True,
        }
    )
    provider.initialize("discord:test", agent_context="primary", platform="discord")
    try:
        provider.sync_turn(
            "I prefer simple status answers before deeper investigation.",
            "Understood.",
            session_id="discord:test",
        )
        provider.sync_turn(
            "Implement the memory cron phases now.",
            "Working on it.",
            session_id="discord:test",
        )
        pending = provider._store.list_events(status="pending")
        assert len(pending) == 1
        payload = json.loads(pending[0]["payload_json"])
        assert payload["source_type"] == "user_explicit"
        assert payload["volatility"] == "stable"
        assert "simple status answers" in payload["evidence"]
    finally:
        provider.shutdown()


def test_non_primary_provider_context_does_not_capture_turns(tmp_path):
    provider = HolographicMemoryProvider(
        config={
            "db_path": str(tmp_path / "subagent.db"),
            "hrr_dim": 64,
            "event_capture": True,
        }
    )
    provider.initialize("subagent:test", agent_context="subagent", platform="cli")
    try:
        provider.sync_turn(
            "I prefer this should never become a user memory.",
            "Ignored.",
            session_id="subagent:test",
        )
        assert provider._store.list_events(status="pending") == []
    finally:
        provider.shutdown()


def test_builtin_memory_write_metadata_is_preserved_without_deep_delete(tmp_path):
    provider = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "bridge.db"), "hrr_dim": 64}
    )
    provider.initialize("discord:test", agent_context="primary", platform="discord")
    assert provider._store is not None
    try:
        metadata = {
            "session_id": "discord:test",
            "platform": "discord",
            "write_origin": "assistant_tool",
            "execution_context": "foreground",
            "tool_call_id": "call-1",
        }
        provider.on_memory_write(
            "add",
            "user",
            "Falko prefers verified operational answers.",
            metadata=metadata,
        )
        facts = provider._store.list_facts(limit=10)
        assert len(facts) == 1
        fact = provider._store.get_fact(facts[0]["fact_id"])
        assert fact is not None
        assert fact["provenance"][0]["session_id"] == "discord:test"
        bridge_events = [e for e in fact["events"] if e["event_type"] == "builtin_memory_add"]
        assert len(bridge_events) == 1
        assert json.loads(bridge_events[0]["metadata_json"])["tool_call_id"] == "call-1"

        provider.on_memory_write(
            "remove",
            "user",
            "",
            metadata={**metadata, "tool_call_id": "call-2", "old_text": "verified operational"},
        )
        assert provider._store.get_fact(facts[0]["fact_id"]) is not None
        removal = provider._store._conn.execute(
            "SELECT metadata_json FROM fact_events WHERE event_type = 'builtin_memory_remove'"
        ).fetchone()
        assert removal is not None
        assert json.loads(removal["metadata_json"])["old_text"] == "verified operational"
        assert provider._store._conn.execute("PRAGMA user_version").fetchone()[0] >= 1
    finally:
        provider.shutdown()


def test_string_false_does_not_enable_legacy_auto_extract(tmp_path):
    provider = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "legacy.db"), "hrr_dim": 64, "auto_extract": "false"}
    )
    provider.initialize("discord:test", agent_context="primary", platform="discord")
    assert provider._store is not None
    try:
        provider.on_session_end([{"role": "user", "content": "I prefer this must not be directly extracted."}])
        assert provider._store.list_facts(limit=10) == []
    finally:
        provider.shutdown()


def test_phase3_tool_handler_adds_provenance_and_resolves_event(tmp_path):
    provider = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "tools.db"), "hrr_dim": 64}
    )
    provider.initialize("discord:test", agent_context="primary", platform="discord")
    assert provider._store is not None
    try:
        event_id = provider._store.enqueue_event(
            "retention_candidate",
            session_id="discord:test",
            payload={"evidence": "I prefer explicit verification."},
        )
        added = json.loads(provider.handle_tool_call("fact_store", {
            "action": "add",
            "content": "Falko prefers explicit verification.",
            "category": "user_pref",
            "entities": ["Falko"],
            "source_type": "user_explicit",
            "source_session_id": "discord:test",
            "evidence": "I prefer explicit verification.",
            "volatility": "stable",
        }))
        assert added["fact"]["provenance"][0]["source_type"] == "user_explicit"
        resolved = json.loads(provider.handle_tool_call("fact_store", {
            "action": "resolve_event",
            "event_id": event_id,
            "event_status": "applied",
            "fact_id": added["fact_id"],
            "resolution_note": "Direct durable preference retained and verified.",
        }))
        assert resolved["event"]["status"] == "applied"
        assert resolved["event"]["resolved_fact_id"] == added["fact_id"]
    finally:
        provider.shutdown()


def test_entity_candidate_filter_rejects_sentence_prefixes_and_symbols(tmp_path):
    store = MemoryStore(tmp_path / "entities.db", hrr_dim=64)
    try:
        extracted = store._extract_entities(
            'For Falko this matters. On Falko this depends. "*" and "Valid Entity".'
        )
        assert "For Falko" not in extracted
        assert "On Falko" not in extracted
        assert "*" not in extracted
        assert "Valid Entity" in extracted
    finally:
        store.close()
