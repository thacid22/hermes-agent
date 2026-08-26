"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import hashlib
import json
import os
import re
import sqlite3
import threading
from pathlib import Path

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    shown_count     INTEGER DEFAULT 0,
    selected_count  INTEGER DEFAULT 0,
    last_retrieved_at TIMESTAMP,
    last_selected_at TIMESTAMP,
    last_helpful_at TIMESTAMP,
    original_category TEXT,
    original_trust_score REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_provenance (
    provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id       INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    source_type   TEXT NOT NULL,
    session_id    TEXT DEFAULT '',
    evidence      TEXT DEFAULT '',
    evidence_hash TEXT NOT NULL,
    volatility    TEXT DEFAULT 'stable',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(fact_id, source_type, session_id, evidence_hash)
);

CREATE TABLE IF NOT EXISTS fact_relations (
    relation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    source_fact_id INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    target_fact_id INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    relation_type  TEXT NOT NULL,
    evidence       TEXT DEFAULT '',
    active         INTEGER DEFAULT 1,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retired_at     TIMESTAMP,
    UNIQUE(source_fact_id, target_fact_id, relation_type)
);

CREATE TABLE IF NOT EXISTS memory_events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type    TEXT NOT NULL,
    session_id    TEXT DEFAULT '',
    payload_json  TEXT NOT NULL,
    payload_hash  TEXT NOT NULL,
    status        TEXT DEFAULT 'pending',
    resolved_fact_id INTEGER REFERENCES facts(fact_id),
    resolution_note TEXT DEFAULT '',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at  TIMESTAMP,
    UNIQUE(event_type, session_id, payload_hash)
);

CREATE TABLE IF NOT EXISTS fact_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id         INTEGER,
    event_type      TEXT NOT NULL,
    before_json     TEXT DEFAULT '',
    after_json      TEXT DEFAULT '',
    metadata_json   TEXT DEFAULT '',
    idempotency_key TEXT UNIQUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_provenance_fact ON fact_provenance(fact_id);
CREATE INDEX IF NOT EXISTS idx_relations_source ON fact_relations(source_fact_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON fact_relations(target_fact_id);
CREATE INDEX IF NOT EXISTS idx_events_status ON memory_events(status, created_at);
CREATE INDEX IF NOT EXISTS idx_fact_events_fact ON fact_events(fact_id, created_at);
"""

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)

_INVALID_ENTITY_PREFIXES = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "if", "in",
    "into", "of", "on", "only", "or", "the", "this", "to", "when", "with",
})
_PROTECTED_FACT_RE = re.compile(
    r"(?i)\b(identity|billybot|safety|privacy|falko prefers|falko wants)\b"
)


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    # Process-wide shared connection registry. Main agents and delegate
    # providers can coexist in one process; one connection + one RLock per
    # resolved DB path prevents independent WAL writers from fighting over the
    # same database. Connections are refcounted so sibling stores remain valid.
    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY

        try:
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with MemoryStore._shared_guard:
            entry = MemoryStore._shared.get(self._key)
            if entry is None:
                conn = sqlite3.connect(
                    self._key,
                    check_same_thread=False,
                    timeout=10.0,
                    # Derived-state repairs and event writes must not leave a
                    # failed transaction holding the process-wide write lock.
                    isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys=ON")
                entry = {
                    "conn": conn,
                    "lock": threading.RLock(),
                    "refs": 0,
                    "ready": False,
                }
                MemoryStore._shared[self._key] = entry
            entry["refs"] += 1
            self._entry = entry
            self._conn = entry["conn"]
            self._lock = entry["lock"]

        with self._lock:
            if not self._entry["ready"]:
                self._init_db()
                self._entry["ready"] = True

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist. Enable WAL mode."""
        # Use the shared WAL-fallback helper so memory_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same issue as
        # state.db / kanban.db — see hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="memory_store.db (holographic)")
        self._conn.executescript(_SCHEMA)
        # Additive migrations for existing profile databases.
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "hrr_vector" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
        migrations = {
            "shown_count": "INTEGER DEFAULT 0",
            "selected_count": "INTEGER DEFAULT 0",
            "last_retrieved_at": "TIMESTAMP",
            "last_selected_at": "TIMESTAMP",
            "last_helpful_at": "TIMESTAMP",
            "original_category": "TEXT",
            "original_trust_score": "REAL",
        }
        for name, sql_type in migrations.items():
            if name not in columns:
                self._conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {sql_type}")
        self._conn.execute(
            "UPDATE facts SET shown_count = retrieval_count "
            "WHERE shown_count = 0 AND retrieval_count > 0"
        )
        relation_columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(fact_relations)").fetchall()
        }
        if "active" not in relation_columns:
            self._conn.execute("ALTER TABLE fact_relations ADD COLUMN active INTEGER DEFAULT 1")
        if "retired_at" not in relation_columns:
            self._conn.execute("ALTER TABLE fact_relations ADD COLUMN retired_at TIMESTAMP")
        self._conn.execute("PRAGMA user_version = 1")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
        entities: list[str] | None = None,
        provenance: dict | None = None,
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id without modifying the row. Extracts entities from
        the content and links them to the fact.
        """
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (content, category, tags, self.default_trust),
                )
                self._conn.commit()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                # Duplicate content — preserve any new links/provenance.
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                fact_id = int(row["fact_id"])
                if entities:
                    self.link_entities(fact_id, entities)
                if provenance:
                    self.add_provenance(fact_id, **provenance)
                self.append_fact_event(
                    fact_id,
                    "fact_reconfirmed",
                    after={"content": content, "category": category, "tags": tags},
                    metadata={"provenance": provenance or {}},
                )
                return fact_id

            # Entity extraction and explicit linking.
            for name in [*self._extract_entities(content), *(entities or [])]:
                if not self._valid_entity_candidate(name):
                    continue
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)

            if provenance:
                self.add_provenance(fact_id, **provenance)

            self.append_fact_event(
                fact_id,
                "fact_added",
                after={"content": content, "category": category, "tags": tags, "trust_score": self.default_trust},
                metadata={"provenance": provenance or {}},
            )
            # HRR vectors/banks are derived state. If this fails, the durable
            # fact/provenance/event remain and the watchdog repairs the drift.
            self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category)

            return fact_id

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            # FTS5 AND-joins tokens by default, which zeroes out recall on
            # natural-language queries. Reuse the retriever's sanitizer
            # (stopword drop + OR-join content tokens). Imported lazily to
            # avoid a store->retrieval import cycle.
            from plugins.memory.holographic.retrieval import FactRetriever

            match_query = FactRetriever._sanitize_fts_query(query)
            params: list = [match_query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = self._conn.execute(sql, params).fetchall()
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"""
                    UPDATE facts
                    SET retrieval_count = retrieval_count + 1,
                        shown_count = shown_count + 1,
                        last_retrieved_at = CURRENT_TIMESTAMP
                    WHERE fact_id IN ({placeholders})
                    """,
                    ids,
                )
                self._conn.commit()

            return results

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            self._conn.commit()

            # If content changed, re-extract entities
            if content is not None:
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                for name in self._extract_entities(content):
                    entity_id = self._resolve_entity(name)
                    self._link_fact_entity(fact_id, entity_id)
                self._conn.commit()

            # Recompute HRR vector if content changed
            if content is not None:
                self._compute_hrr_vector(fact_id, content)
            # Rebuild banks for both sides of a category move. Rebuilding only
            # the destination leaves the source bank's fact_count/vector stale.
            old_category = row["category"]
            new_category = category or old_category
            self._rebuild_bank(new_category)
            if old_category != new_category:
                self._rebuild_bank(old_category)
            after_row = self._conn.execute(
                "SELECT content, category, tags, trust_score FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            self.append_fact_event(
                fact_id,
                "fact_updated",
                before=dict(row),
                after=dict(after_row),
            )

            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links. Returns True if the row existed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, content, category, tags, trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self.append_fact_event(
                    fact_id,
                    "fact_removed",
                    before=dict(row),
                    metadata={"irreversible": True},
                    commit=False,
                )
                # Resolved curation events are historical evidence, not owners
                # of the fact. Detach them before deletion while preserving the
                # event and its resolution status/note.
                self._conn.execute(
                    "UPDATE memory_events SET resolved_fact_id = NULL "
                    "WHERE resolved_fact_id = ?",
                    (fact_id,),
                )
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            self._rebuild_bank(row["category"])
            return True

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        with self._lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_fact(self, fact_id: int) -> dict | None:
        """Return one fact with provenance, relations, and explicit entities."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return None
            fact = dict(row)
            fact.pop("hrr_vector", None)
            fact["entities"] = [
                item["name"]
                for item in self._conn.execute(
                    """
                    SELECT e.name FROM entities e
                    JOIN fact_entities fe ON fe.entity_id = e.entity_id
                    WHERE fe.fact_id = ? ORDER BY e.name COLLATE NOCASE
                    """,
                    (fact_id,),
                ).fetchall()
            ]
            fact["provenance"] = [
                dict(item)
                for item in self._conn.execute(
                    "SELECT * FROM fact_provenance WHERE fact_id = ? ORDER BY provenance_id",
                    (fact_id,),
                ).fetchall()
            ]
            fact["relations"] = [
                dict(item)
                for item in self._conn.execute(
                    """
                    SELECT * FROM fact_relations
                    WHERE source_fact_id = ? OR target_fact_id = ?
                    ORDER BY relation_id
                    """,
                    (fact_id, fact_id),
                ).fetchall()
            ]
            fact["events"] = self.list_fact_events(fact_id)
            return fact

    def append_fact_event(
        self,
        fact_id: int | None,
        event_type: str,
        *,
        before: dict | None = None,
        after: dict | None = None,
        metadata: dict | None = None,
        idempotency_key: str | None = None,
        commit: bool = True,
    ) -> dict:
        """Append an auditable mutation event without requiring the fact to remain live."""
        def encode(value: dict | None) -> str:
            return json.dumps(value or {}, sort_keys=True, ensure_ascii=False, default=str)

        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO fact_events
                    (fact_id, event_type, before_json, after_json, metadata_json, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_id,
                    event_type,
                    encode(before),
                    encode(after),
                    encode(metadata),
                    idempotency_key,
                ),
            )
            if commit:
                self._conn.commit()
            if idempotency_key:
                row = self._conn.execute(
                    "SELECT * FROM fact_events WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM fact_events ORDER BY event_id DESC LIMIT 1"
                ).fetchone()
            return dict(row)

    def list_fact_events(self, fact_id: int, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM fact_events WHERE fact_id = ? ORDER BY event_id LIMIT ?",
                (fact_id, max(1, min(int(limit), 1000))),
            ).fetchall()
            return [dict(row) for row in rows]

    def add_provenance(
        self,
        fact_id: int,
        source_type: str,
        session_id: str = "",
        evidence: str = "",
        volatility: str = "stable",
    ) -> dict:
        """Attach auditable source evidence to a fact, idempotently."""
        if source_type not in {"user_explicit", "user_correction", "assistant_inferred", "live_verified", "archive", "manual"}:
            raise ValueError(f"unsupported source_type: {source_type}")
        if volatility not in {"stable", "mutable", "ephemeral"}:
            raise ValueError(f"unsupported volatility: {volatility}")
        evidence = str(evidence or "").strip()[:2000]
        evidence_hash = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
        with self._lock:
            if self._conn.execute("SELECT 1 FROM facts WHERE fact_id = ?", (fact_id,)).fetchone() is None:
                raise KeyError(f"fact_id {fact_id} not found")
            self._conn.execute(
                """
                INSERT OR IGNORE INTO fact_provenance
                    (fact_id, source_type, session_id, evidence, evidence_hash, volatility)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fact_id, source_type, session_id or "", evidence, evidence_hash, volatility),
            )
            self._conn.commit()
            row = self._conn.execute(
                """
                SELECT * FROM fact_provenance
                WHERE fact_id = ? AND source_type = ? AND session_id = ? AND evidence_hash = ?
                """,
                (fact_id, source_type, session_id or "", evidence_hash),
            ).fetchone()
            return dict(row)

    def add_relation(
        self,
        source_fact_id: int,
        target_fact_id: int,
        relation_type: str,
        evidence: str = "",
    ) -> dict:
        """Record a typed, reversible relation between two existing facts."""
        if relation_type not in {"duplicate_of", "superseded_by", "supports", "contradicts"}:
            raise ValueError(f"unsupported relation_type: {relation_type}")
        if source_fact_id == target_fact_id:
            raise ValueError("a fact cannot relate to itself")
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE fact_id IN (?, ?)",
                (source_fact_id, target_fact_id),
            ).fetchone()[0]
            if count != 2:
                raise KeyError("source or target fact does not exist")
            self._conn.execute(
                """
                INSERT INTO fact_relations
                    (source_fact_id, target_fact_id, relation_type, evidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_fact_id, target_fact_id, relation_type)
                DO UPDATE SET evidence = excluded.evidence, active = 1, retired_at = NULL
                """,
                (source_fact_id, target_fact_id, relation_type, str(evidence or "")[:2000]),
            )
            self._conn.commit()
            row = self._conn.execute(
                """
                SELECT * FROM fact_relations
                WHERE source_fact_id = ? AND target_fact_id = ? AND relation_type = ?
                """,
                (source_fact_id, target_fact_id, relation_type),
            ).fetchone()
            return dict(row)

    def quarantine_fact(
        self,
        fact_id: int,
        *,
        related_fact_id: int,
        relation_type: str,
        evidence: str,
        allow_protected: bool = False,
    ) -> dict:
        """Reversibly isolate a duplicate/superseded fact; never deletes content."""
        with self._lock:
            row = self._conn.execute(
                "SELECT content, category, tags, trust_score, original_category, original_trust_score FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")
            if related_fact_id == fact_id:
                raise ValueError("canonical/superseding fact must be different")
            if self._conn.execute(
                "SELECT 1 FROM facts WHERE fact_id = ? AND category != 'quarantine'",
                (related_fact_id,),
            ).fetchone() is None:
                raise ValueError("related fact must exist and remain active")
            if not evidence or len(evidence.strip()) < 12:
                raise ValueError("quarantine requires concrete evidence")
            protected = row["category"] == "user_pref" or bool(_PROTECTED_FACT_RE.search(row["content"]))
            if protected and not allow_protected:
                raise ValueError("protected fact requires direct correction evidence and explicit override")

            original_category = row["original_category"] or row["category"]
            original_trust = row["original_trust_score"]
            if original_trust is None:
                original_trust = float(row["trust_score"])
            tags = {tag.strip() for tag in str(row["tags"] or "").split(",") if tag.strip()}
            tags.update({"dream-quarantine", f"original-category:{original_category}"})
            if relation_type == "duplicate_of":
                tags.add(f"duplicate-of:{related_fact_id}")
            elif relation_type == "superseded_by":
                tags.add(f"superseded-by:{related_fact_id}")
            else:
                raise ValueError("quarantine relation must be duplicate_of or superseded_by")

            self._conn.execute(
                """
                UPDATE facts
                SET category = 'quarantine', tags = ?, original_category = ?,
                    original_trust_score = ?, trust_score = MIN(trust_score, 0.2),
                    updated_at = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (",".join(sorted(tags)), original_category, original_trust, fact_id),
            )
            self._conn.commit()
            relation = self.add_relation(fact_id, related_fact_id, relation_type, evidence)
            self._rebuild_bank(original_category)
            self._rebuild_bank("quarantine")
            self.append_fact_event(
                fact_id,
                "fact_quarantined",
                before=dict(row),
                after={
                    "category": "quarantine",
                    "original_category": original_category,
                    "original_trust_score": original_trust,
                    "related_fact_id": related_fact_id,
                    "relation_type": relation_type,
                },
                metadata={"evidence": evidence},
            )
            return {"fact": self.get_fact(fact_id), "relation": relation}

    def restore_fact(self, fact_id: int, *, evidence: str) -> dict:
        """Restore a quarantined fact and retire its active quarantine relations."""
        if not evidence or len(evidence.strip()) < 12:
            raise ValueError("restore requires concrete evidence")
        with self._lock:
            row = self._conn.execute(
                """
                SELECT content, category, tags, trust_score,
                       original_category, original_trust_score
                FROM facts WHERE fact_id = ?
                """,
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")
            if row["category"] != "quarantine" or not row["original_category"]:
                raise ValueError("fact is not reversibly quarantined")
            restored_category = str(row["original_category"])
            restored_trust = (
                float(row["original_trust_score"])
                if row["original_trust_score"] is not None
                else max(float(row["trust_score"]), self.default_trust)
            )
            tags = {
                tag.strip()
                for tag in str(row["tags"] or "").split(",")
                if tag.strip()
                and tag.strip() != "dream-quarantine"
                and not tag.strip().startswith(("original-category:", "duplicate-of:", "superseded-by:"))
            }
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    UPDATE facts
                    SET category = ?, tags = ?, trust_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE fact_id = ?
                    """,
                    (restored_category, ",".join(sorted(tags)), restored_trust, fact_id),
                )
                self._conn.execute(
                    """
                    UPDATE fact_relations
                    SET active = 0, retired_at = CURRENT_TIMESTAMP
                    WHERE source_fact_id = ? AND active = 1
                      AND relation_type IN ('duplicate_of', 'superseded_by')
                    """,
                    (fact_id,),
                )
                after = {
                    "category": restored_category,
                    "tags": ",".join(sorted(tags)),
                    "trust_score": restored_trust,
                }
                self.append_fact_event(
                    fact_id,
                    "fact_restored",
                    before=dict(row),
                    after=after,
                    metadata={"evidence": evidence},
                    commit=False,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            self._rebuild_bank("quarantine")
            self._rebuild_bank(restored_category)
            return {"restored": True, "fact": self.get_fact(fact_id)}

    def enqueue_event(
        self,
        event_type: str,
        *,
        session_id: str = "",
        payload: dict,
    ) -> int:
        """Queue a profile-local curation event, deduplicated by canonical payload."""
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO memory_events
                    (event_type, session_id, payload_json, payload_hash)
                VALUES (?, ?, ?, ?)
                """,
                (event_type, session_id or "", payload_json, payload_hash),
            )
            self._conn.commit()
            row = self._conn.execute(
                """
                SELECT event_id FROM memory_events
                WHERE event_type = ? AND session_id = ? AND payload_hash = ?
                """,
                (event_type, session_id or "", payload_hash),
            ).fetchone()
            return int(row["event_id"])

    def list_events(self, status: str = "pending", limit: int = 100) -> list[dict]:
        """List curation events without changing retrieval telemetry."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memory_events WHERE status = ?
                ORDER BY created_at, event_id LIMIT ?
                """,
                (status, max(1, min(int(limit), 1000))),
            ).fetchall()
            return [dict(row) for row in rows]

    def resolve_event(
        self,
        event_id: int,
        status: str,
        *,
        fact_id: int | None = None,
        note: str = "",
    ) -> dict:
        """Resolve a queued event as applied/rejected/deferred/error."""
        if status not in {"applied", "rejected", "deferred", "error"}:
            raise ValueError(f"unsupported event status: {status}")
        with self._lock:
            self._conn.execute(
                """
                UPDATE memory_events
                SET status = ?, resolved_fact_id = ?, resolution_note = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE event_id = ?
                """,
                (status, fact_id, str(note or "")[:2000], event_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM memory_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"event_id {event_id} not found")
            return dict(row)

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count, selected_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            self._conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    selected_count = selected_count + 1,
                    last_selected_at = CURRENT_TIMESTAMP,
                    last_helpful_at = CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE last_helpful_at END,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, helpful_increment, fact_id),
            )
            self._conn.commit()
            self.append_fact_event(
                fact_id,
                "fact_feedback",
                before={
                    "trust_score": old_trust,
                    "helpful_count": int(row["helpful_count"]),
                    "selected_count": int(row["selected_count"]),
                },
                after={
                    "trust_score": new_trust,
                    "helpful_count": int(row["helpful_count"]) + helpful_increment,
                    "selected_count": int(row["selected_count"]) + 1,
                },
                metadata={"helpful": helpful},
            )

            return {
                "fact_id":      fact_id,
                "old_trust":    old_trust,
                "new_trust":    new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_entity_candidate(name: str) -> bool:
        stripped = " ".join(str(name or "").split()).strip(".,;:!?\"'()[]{}")
        if len(stripped) < 2 or len(stripped) > 100:
            return False
        if not any(ch.isalnum() for ch in stripped):
            return False
        first = stripped.split()[0].lower()
        if first in _INVALID_ENTITY_PREFIXES:
            return False
        return True

    def link_entities(self, fact_id: int, names: list[str], *, replace: bool = False) -> list[str]:
        """Link explicit entity names to a fact and rebuild its HRR vector."""
        with self._lock:
            row = self._conn.execute(
                "SELECT content, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")
            if replace:
                self._conn.execute("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
            linked: list[str] = []
            for raw_name in names:
                name = " ".join(str(raw_name or "").split()).strip()
                if not self._valid_entity_candidate(name):
                    continue
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)
                if name.lower() not in {item.lower() for item in linked}:
                    linked.append(name)
            self._conn.commit()
            self._compute_hrr_vector(fact_id, row["content"])
            self._rebuild_bank(row["category"])
            return linked

    def resolve_entity_id(self, name: str) -> int | None:
        """Resolve an existing entity by exact name or comma-delimited alias."""
        with self._lock:
            row = self._conn.execute(
                "SELECT entity_id FROM entities WHERE name = ? COLLATE NOCASE",
                (name.strip(),),
            ).fetchone()
            if row is not None:
                return int(row["entity_id"])
            wanted = name.strip().lower()
            for candidate in self._conn.execute(
                "SELECT entity_id, aliases FROM entities WHERE aliases != ''"
            ).fetchall():
                aliases = {item.strip().lower() for item in str(candidate["aliases"]).split(",") if item.strip()}
                if wanted in aliases:
                    return int(candidate["entity_id"])
            return None

    def fact_ids_for_entity(
        self,
        name: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
    ) -> list[int]:
        """Return trusted active fact IDs explicitly linked to an entity."""
        entity_id = self.resolve_entity_id(name)
        if entity_id is None:
            return []
        params: list = [entity_id, min_trust]
        category_sql = (
            "AND f.category != 'quarantine' "
            "AND f.tags NOT LIKE '%dream-quarantine%' "
            "AND f.tags NOT LIKE '%quarantine%'"
        )
        if category:
            category_sql += " AND f.category = ?"
            params.append(category)
        rows = self._conn.execute(
            f"""
            SELECT f.fact_id FROM facts f
            JOIN fact_entities fe ON fe.fact_id = f.fact_id
            WHERE fe.entity_id = ? AND f.trust_score >= ? {category_sql}
            ORDER BY f.fact_id
            """,
            params,
        ).fetchall()
        return [int(row["fact_id"]) for row in rows]

    def enrich_entity_links(self, names: list[str]) -> dict:
        """One-time/idempotent enrichment using an explicit policy seed list."""
        valid_names = [" ".join(str(name).split()).strip() for name in names]
        valid_names = [name for name in valid_names if self._valid_entity_candidate(name)]
        linked = 0
        touched: set[int] = set()
        with self._lock:
            facts = self._conn.execute("SELECT fact_id, content, category FROM facts").fetchall()
            for fact in facts:
                content = str(fact["content"])
                matched = [
                    name for name in valid_names
                    if re.search(rf"(?i)(?<!\w){re.escape(name)}(?!\w)", content)
                ]
                for name in matched:
                    entity_id = self._resolve_entity(name)
                    before = self._conn.total_changes
                    self._link_fact_entity(int(fact["fact_id"]), entity_id)
                    if self._conn.total_changes > before:
                        linked += 1
                        touched.add(int(fact["fact_id"]))
            self._conn.commit()
            for fact_id in touched:
                row = self._conn.execute(
                    "SELECT content, category FROM facts WHERE fact_id = ?", (fact_id,)
                ).fetchone()
                self._compute_hrr_vector(fact_id, row["content"])
            for category in {
                self._conn.execute("SELECT category FROM facts WHERE fact_id = ?", (fid,)).fetchone()[0]
                for fid in touched
            }:
                self._rebuild_bank(category)
        return {"links_added": linked, "facts_touched": len(touched), "seed_count": len(valid_names)}

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using simple regex rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Double-quoted terms             e.g. "Python"
        3. Single-quoted terms             e.g. 'pytest'
        4. AKA patterns                    e.g. "Guido aka BDFL" -> two entities

        Returns a deduplicated list preserving first-seen order.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = " ".join(name.split()).strip()
            if not self._valid_entity_candidate(stripped):
                return
            if stripped.lower() not in seen:
                seen.add(stripped.lower())
                candidates.append(stripped)

        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        return candidates

    def _resolve_entity(self, name: str) -> int:
        """Find an existing entity by exact name/alias or create one."""
        name = " ".join(str(name or "").split()).strip()
        if not self._valid_entity_candidate(name):
            raise ValueError(f"invalid entity candidate: {name!r}")
        entity_id = self.resolve_entity_id(name)
        if entity_id is not None:
            return entity_id

        cur = self._conn.execute(
            "INSERT INTO entities (name) VALUES (?)", (name,)
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[return-value]

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        self._conn.commit()

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        with self._lock:
            if not self._hrr_available:
                return

            # Get entities linked to this fact
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

            vector = hrr.encode_fact(content, entities, self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        with self._lock:
            if not self._hrr_available:
                return

            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

            if not rows:
                self._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
                self._conn.commit()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"], dim=self.hrr_dim) for row in rows]
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

            # Check SNR
            hrr.snr_estimate(self.hrr_dim, fact_count)

            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._conn.commit()

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.
        """
        with self._lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    @classmethod
    def release_all_under(cls, directory: "str | Path") -> int:
        """Force-close every shared connection whose database lives under ``directory``.

        ``close()`` is refcount-driven, so a live holder (e.g. an agent's
        memory provider) keeps a profile's SQLite handle open indefinitely.
        That is exactly what a profile delete must break on Windows: the
        desktop's main ``serve`` process opens ``memory_store.db`` for every
        known profile, and ``rmtree`` of the profile directory fails with
        ``WinError 32`` while any of those handles is open (#88347). This
        closes the matching connections unconditionally — the directory is
        going away, so later use by a stale holder is expected to fail — and
        returns how many were closed. In a process that holds none (e.g. the
        CLI deleting from outside serve) this is a harmless no-op returning 0.
        """
        root = os.path.normcase(str(Path(directory).expanduser().resolve())) + os.sep
        with cls._shared_guard:
            # Snapshot the keys first so the registry stays stable while
            # connections are closed inside their per-database locks (closing
            # can run no user code, but this keeps the invariant obvious).
            doomed = [
                key
                for key in cls._shared
                if os.path.normcase(key).startswith(root)
            ]
            for key in doomed:
                entry = cls._shared.pop(key)
                try:
                    with entry["lock"]:
                        entry["conn"].close()
                except Exception:
                    # A connection that is already closed or broken must not
                    # abort releasing its siblings.
                    pass
        return len(doomed)

    def close(self) -> None:
        """Release this instance's reference to the shared connection."""
        if getattr(self, "_entry", None) is None:
            return
        with MemoryStore._shared_guard:
            entry = self._entry
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                try:
                    entry["conn"].close()
                finally:
                    # Pop only OUR entry. After release_all_under() force-
                    # closed this entry (profile delete, #88347) a same-path
                    # store may have re-registered a FRESH entry under the
                    # same key; a stale holder's late close() must not evict
                    # it — that would silently reintroduce the multi-writer
                    # contention this registry exists to prevent.
                    if MemoryStore._shared.get(self._key) is entry:
                        MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
