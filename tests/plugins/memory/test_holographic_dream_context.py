"""Regression tests for the local Holographic dream watchdog/curator script."""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.store import MemoryStore


SCRIPT = Path("/home/falko/.hermes/scripts/memory/holographic_dream_context.py")
ACTIVATION_SCRIPT = Path("/home/falko/.hermes/scripts/memory/holographic_activate_after_turn.py")
if not SCRIPT.is_file() or not ACTIVATION_SCRIPT.is_file():
    pytest.skip(
        "local Holographic dream scripts are not installed",
        allow_module_level=True,
    )

SPEC = importlib.util.spec_from_file_location("holographic_dream_context", SCRIPT)
assert SPEC and SPEC.loader
DREAM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DREAM)

ACTIVATION_SPEC = importlib.util.spec_from_file_location(
    "holographic_activate_after_turn", ACTIVATION_SCRIPT
)
assert ACTIVATION_SPEC and ACTIVATION_SPEC.loader
ACTIVATION = importlib.util.module_from_spec(ACTIVATION_SPEC)
ACTIVATION_SPEC.loader.exec_module(ACTIVATION)


def test_activation_schema_parser_accepts_native_and_wrapped_shapes():
    body = {
        "name": "fact_store",
        "parameters": {
            "properties": {"action": {"enum": ["add", "restore"]}}
        },
    }
    assert ACTIVATION.fact_store_actions([body]) == {"add", "restore"}
    assert ACTIVATION.fact_store_actions([{"type": "function", "function": body}]) == {
        "add",
        "restore",
    }


def test_backup_is_integrity_checked_and_not_duplicated_into_run_dir(tmp_path, monkeypatch):
    source = tmp_path / "live.db"
    with MemoryStore(source, hrr_dim=64) as store:
        store.add_fact("A durable test fact.")
    backup_root = tmp_path / "backups"
    run_dir = tmp_path / "run"
    monkeypatch.setattr(DREAM, "BACKUP_ROOT", backup_root)

    result = DREAM.backup_database(source, run_dir, "20260717-010000")

    assert result["ok"] is True
    assert result["integrity_check"] == ["ok"]
    assert Path(result["path"]).exists()
    assert "run_copy" not in result
    assert list(run_dir.glob("*.db")) == []


def test_backup_retention_keeps_recent_weekly_monthly_generations(tmp_path, monkeypatch):
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    monkeypatch.setattr(DREAM, "BACKUP_ROOT", backup_root)
    # 40 deterministic daily generations across several ISO weeks/months.
    for day in range(1, 41):
        stamp = (DREAM.dt.datetime(2026, 1, 1) + DREAM.dt.timedelta(days=day)).strftime("%Y%m%d-010000")
        (backup_root / f"memory_store-{stamp}.db").write_bytes(b"db")

    result = DREAM.prune_backup_generations(
        daily=14, weekly=8, monthly=12, dry_run=False
    )

    remaining = list(backup_root.glob("memory_store-*.db"))
    assert result["deleted"] > 0
    assert len(remaining) == result["kept"]
    assert len(remaining) >= 14
    assert len(remaining) < 40


def test_temporary_word_is_only_a_weak_signal_for_protected_preference(tmp_path):
    db = tmp_path / "memory.db"
    with MemoryStore(db, hrr_dim=64) as store:
        protected_id = store.add_fact(
            "Falko prefers temporary files outside repositories.",
            category="user_pref",
            entities=["Falko"],
        )
        transient_id = store.add_fact(
            "Current run helper is PID 1234 under /tmp/ and can be removed again.",
            category="general",
        )

    audit = DREAM.inspect_database(db)
    candidate_ids = {row["fact_id"] for row in audit["quarantine_candidates"]}
    protected = next(row for row in audit["classifications"] if row["fact_id"] == protected_id)

    assert protected_id not in candidate_ids
    assert "transient-signal" in protected["classes"]
    assert transient_id in candidate_ids


def test_retrieval_evaluation_uses_backup_without_incrementing_counts(tmp_path):
    db = tmp_path / "memory.db"
    with MemoryStore(db, hrr_dim=64) as store:
        fact_id = store.add_fact(
            "Falko prefers bounded triage and an early useful answer.",
            category="user_pref",
            entities=["Falko"],
        )
    result = DREAM.run_retrieval_evaluation(
        db,
        [{"name": "triage", "action": "search", "query": "bounded triage", "expected_fact_ids": [fact_id]}],
    )
    assert result["ok"] is True
    assert result["passed"] == 1
    with MemoryStore(db, hrr_dim=64) as store:
        fact = store.get_fact(fact_id)
        assert fact["retrieval_count"] == 0
        assert fact["shown_count"] == 0


def test_legacy_retriever_contract_reports_drift_without_crashing(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    with MemoryStore(db, hrr_dim=64) as store:
        fact_id = store.add_fact("A retrieval contract fixture.")

    class LegacyStore:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    class LegacyRetriever:
        def __init__(self, store, hrr_dim):
            self.store = store
            self.hrr_dim = hrr_dim

        def search(self, _query, limit=20):
            return [{"fact_id": fact_id}][:limit]

    monkeypatch.setattr(
        DREAM,
        "_load_holographic_runtime",
        lambda: (LegacyStore, LegacyRetriever),
    )
    result = DREAM.run_retrieval_evaluation(
        db,
        [{"name": "legacy", "action": "search", "query": "fixture", "expected_fact_ids": [fact_id]}],
    )

    assert result["contracts_ok"] is True
    assert result["ok"] is False
    assert "track_usage=False" in result["compatibility_warning"]


def test_active_safe_hygiene_repairs_orphan_provenance_rows(tmp_path):
    db = tmp_path / "foreign-keys.db"
    with MemoryStore(db, hrr_dim=64) as store:
        fact_id = store.add_fact(
            "A fact with source provenance.",
            provenance={
                "source_type": "manual",
                "session_id": "test",
                "evidence": "test fixture",
                "volatility": "stable",
            },
        )

    # Reproduce the drift caused by a legacy connection that did not enable
    # SQLite foreign keys: the fact disappears but its provenance child stays.
    con = sqlite3.connect(db)
    try:
        con.execute("PRAGMA foreign_keys=OFF")
        con.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
        con.commit()
    finally:
        con.close()

    before = DREAM.inspect_database(db)
    assert before["foreign_key_violations"]

    result = DREAM.apply_active_safe_hygiene(
        db,
        {
            "mode": "active-safe",
            "allow_memory_mutation": True,
            "allow_mechanical_db_hygiene": True,
            "allow_fact_deletion": False,
            "allow_entity_enrichment": False,
            "allow_hrr_rebuild": False,
        },
        {"ok": True},
    )

    assert result["errors"] == []
    assert any(
        change["kind"] == "orphan_fact_provenance_removed"
        for change in result["applied_changes"]
    )
    assert DREAM.inspect_database(db)["foreign_key_violations"] == []


def test_change_fingerprint_ignores_run_timestamps_but_tracks_findings():
    left = {
        "run_started": "2026-07-17T01:00:00+10:00",
        "audit": {"facts": [{"fact_id": 1, "content_hash": "abc"}], "integrity_check": ["ok"]},
        "pending_events": [],
    }
    right = {
        "run_started": "2026-07-18T01:00:00+10:00",
        "audit": {"facts": [{"fact_id": 1, "content_hash": "abc"}], "integrity_check": ["ok"]},
        "pending_events": [],
    }
    changed = {
        **right,
        "pending_events": [{"event_id": 2, "payload_hash": "new"}],
    }
    assert DREAM.meaningful_fingerprint(left) == DREAM.meaningful_fingerprint(right)
    assert DREAM.meaningful_fingerprint(right) != DREAM.meaningful_fingerprint(changed)
