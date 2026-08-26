"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
        track_usage: bool = True,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim
        self.track_usage = track_usage

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional): decay = 0.5^(age_days / half_life)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        if not candidates:
            return []

        # Stage 2: Rerank with Jaccard + trust + optional decay
        query_tokens = self._tokenize(query)
        # The query vector is loop-invariant — encode it at most once, on
        # the first candidate that actually carries an HRR vector. Lazy on
        # purpose: migrated stores can have FTS candidates whose hrr_vector
        # was never backfilled (MemoryStore._init_db adds the column
        # without backfilling), and those must not pay for an encode
        # nothing will use. encode_text is deterministic (SHA-256 counter
        # blocks), so the hoisted vector is bit-identical to what the
        # per-candidate calls produced.
        query_vec = None
        scored = []

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"], dim=self.hrr_dim)
                if query_vec is None:
                    query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            else:
                hrr_sim = 0.5  # neutral

            # Combine FTS5 + Jaccard + HRR
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim)

            # Trust weighting
            score = relevance * fact["trust_score"]

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        # Strip raw HRR bytes — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)
        self._record_retrievals(results)
        return results

    def probe(
        self,
        entity: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Return facts explicitly linked to an entity, then HRR-rerank them."""
        ids = self.store.fact_ids_for_entity(
            entity, category=category, min_trust=min_trust
        )
        if not ids:
            # Unresolved entities retain a keyword fallback, but never receive
            # arbitrary vector-neighbour results presented as entity recall.
            return self.search(
                entity, category=category, min_trust=min_trust, limit=limit
            )
        results = self._facts_by_ids(ids, query=entity, limit=limit)
        self._record_retrievals(results)
        return results

    def related(
        self,
        entity: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Return graph-adjacent facts through entities shared with seed facts."""
        seed_ids = self.store.fact_ids_for_entity(
            entity, category=category, min_trust=min_trust
        )
        if not seed_ids:
            return self.search(
                entity, category=category, min_trust=min_trust, limit=limit
            )
        placeholders = ",".join("?" for _ in seed_ids)
        entity_rows = self.store._conn.execute(
            f"SELECT DISTINCT entity_id FROM fact_entities WHERE fact_id IN ({placeholders})",
            seed_ids,
        ).fetchall()
        entity_ids = [int(row["entity_id"]) for row in entity_rows]
        if not entity_ids:
            return self.probe(entity, category=category, min_trust=min_trust, limit=limit)
        entity_placeholders = ",".join("?" for _ in entity_ids)
        params: list = [*entity_ids, min_trust]
        category_sql = (
            "AND f.category != 'quarantine' "
            "AND f.tags NOT LIKE '%dream-quarantine%' "
            "AND f.tags NOT LIKE '%quarantine%'"
        )
        if category:
            category_sql += " AND f.category = ?"
            params.append(category)
        rows = self.store._conn.execute(
            f"""
            SELECT f.fact_id, COUNT(DISTINCT fe.entity_id) AS shared_entities
            FROM facts f
            JOIN fact_entities fe ON fe.fact_id = f.fact_id
            WHERE fe.entity_id IN ({entity_placeholders})
              AND f.trust_score >= ? {category_sql}
            GROUP BY f.fact_id
            ORDER BY shared_entities DESC, f.trust_score DESC, f.fact_id
            """,
            params,
        ).fetchall()
        scores = {int(row["fact_id"]): int(row["shared_entities"]) for row in rows}
        results = self._facts_by_ids(list(scores), query=entity, limit=limit)
        for fact in results:
            fact["shared_entities"] = scores.get(int(fact["fact_id"]), 0)
            fact["score"] *= max(1, fact["shared_entities"])
        results.sort(key=lambda row: row["score"], reverse=True)
        results = results[:limit]
        self._record_retrievals(results)
        return results

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Return the explicit AND-intersection of facts linked to every entity."""
        if not entities:
            return []
        linked_sets = [
            set(self.store.fact_ids_for_entity(
                entity, category=category, min_trust=min_trust
            ))
            for entity in entities
        ]
        if any(not ids for ids in linked_sets):
            # If entity resolution is incomplete, do not fabricate a vector JOIN.
            return []
        fact_ids = sorted(set.intersection(*linked_sets))
        if not fact_ids:
            return []
        results = self._facts_by_ids(
            fact_ids, query=" ".join(entities), limit=limit
        )
        self._record_retrievals(results)
        return results

    def contradict(
        self,
        category: str | None = None,
        threshold: float = 0.3,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.

        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        conn = self.store._conn

        # Get all trusted facts with vectors and their linked entities
        where = "WHERE f.hrr_vector IS NOT NULL AND f.trust_score >= ?"
        params: list = [min_trust]
        where += " AND f.category != 'quarantine'"
        where += " AND f.tags NOT LIKE '%dream-quarantine%' AND f.tags NOT LIKE '%quarantine%'"
        if category:
            where += " AND f.category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.created_at, f.updated_at, f.hrr_vector
            FROM facts f
            {where}
            """,
            params,
        ).fetchall()

        if len(rows) < 2:
            return []

        # Guard against O(n²) explosion on large fact stores.
        # At 500 facts, that's ~125K comparisons — acceptable.
        # Above that, only check the most recently updated facts.
        _MAX_CONTRADICT_FACTS = 500
        if len(rows) > _MAX_CONTRADICT_FACTS:
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)
            rows = rows[:_MAX_CONTRADICT_FACTS]

        # Build entity sets per fact
        fact_entities: dict[int, set[str]] = {}
        for row in rows:
            fid = row["fact_id"]
            entity_rows = conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fid,),
            ).fetchall()
            fact_entities[fid] = {r["name"].lower() for r in entity_rows}

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        facts = [dict(r) for r in rows]
        contradictions = []

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                ents1 = fact_entities.get(f1["fact_id"], set())
                ents2 = fact_entities.get(f2["fact_id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0

                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap to be contradictory

                # Content similarity via HRR vectors
                v1 = hrr.bytes_to_phases(f1["hrr_vector"], dim=self.hrr_dim)
                v2 = hrr.bytes_to_phases(f2["hrr_vector"], dim=self.hrr_dim)
                content_sim = hrr.similarity(v1, v2)

                # High entity overlap + low content similarity = potential contradiction
                # contradiction_score: higher = more contradictory
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)

                if contradiction_score >= threshold:
                    # Strip hrr_vector from output (not JSON serializable)
                    f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                    f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                    contradictions.append({
                        "fact_a": f1_clean,
                        "fact_b": f2_clean,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    def _facts_by_ids(
        self,
        fact_ids: list[int],
        *,
        query: str,
        limit: int,
    ) -> list[dict]:
        """Load a grounded candidate set and use HRR only for reranking."""
        if not fact_ids:
            return []
        placeholders = ",".join("?" for _ in fact_ids)
        rows = self.store._conn.execute(
            f"SELECT * FROM facts WHERE fact_id IN ({placeholders})",
            fact_ids,
        ).fetchall()
        query_vec = hrr.encode_text(query, self.hrr_dim) if hrr._HAS_NUMPY else None
        query_tokens = self._tokenize(query)
        results: list[dict] = []
        for row in rows:
            fact = dict(row)
            raw_vector = fact.pop("hrr_vector", None)
            if query_vec is not None and raw_vector:
                similarity = (hrr.similarity(query_vec, hrr.bytes_to_phases(raw_vector, dim=self.hrr_dim)) + 1.0) / 2.0
            else:
                similarity = self._jaccard_similarity(
                    query_tokens, self._tokenize(fact.get("content", ""))
                )
            fact["score"] = (0.5 + 0.5 * similarity) * float(fact["trust_score"])
            results.append(fact)
        results.sort(key=lambda row: (row["score"], row["fact_id"]), reverse=True)
        return results[:limit]

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Score trusted facts by similarity to a target vector."""
        conn = self.store._conn

        where = "WHERE hrr_vector IS NOT NULL AND trust_score >= ?"
        params: list = [min_trust]
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)
            sim = hrr.similarity(target_vec, fact_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _record_retrievals(self, results: list[dict]) -> None:
        """Persist retrieval telemetry for every public recall path.

        The quarantine ledger relies on retrieval_count. Historically only the
        lower-level MemoryStore search updated it, while FactRetriever bypassed
        that method, leaving every fact at zero despite real use.
        """
        if not self.track_usage:
            return
        ids = sorted({int(fact["fact_id"]) for fact in results if fact.get("fact_id") is not None})
        if not ids:
            return
        try:
            with self.store._lock:
                self.store._conn.executemany(
                    """
                    UPDATE facts
                    SET retrieval_count = retrieval_count + 1,
                        shown_count = shown_count + 1,
                        last_retrieved_at = CURRENT_TIMESTAMP
                    WHERE fact_id = ?
                    """,
                    [(fact_id,) for fact_id in ids],
                )
                self.store._conn.commit()
            for fact in results:
                if fact.get("fact_id") in ids:
                    fact["retrieval_count"] = int(fact.get("retrieval_count") or 0) + 1
        except Exception:
            try:
                self.store._conn.rollback()
            except Exception:
                pass

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        # Build query - FTS5 rank is negative (lower = better match)
        # We need to join facts_fts with facts to get all columns
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        # FTS5 defaults to AND-between-tokens, which kills recall on
        # natural-language queries ("what happened with the deployment
        # rollback"). Sanitize: drop stopwords, OR-join content tokens, so
        # any significant term can match.
        params.append(self._sanitize_fts_query(query))

        where_clauses.append("f.category != 'quarantine'")
        where_clauses.append("f.tags NOT LIKE '%dream-quarantine%'")
        where_clauses.append("f.tags NOT LIKE '%quarantine%'")
        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH can fail on malformed queries — fall back to empty
            return []

        if not rows:
            return []

        # Normalize FTS5 rank: rank is negative, lower = better
        # Convert to positive score in [0, 1] range
        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)  # avoid div by zero

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank  # normalize to [0, 1]
            results.append(fact)

        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        # Split on whitespace, lowercase, strip punctuation
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    # Stopwords dropped before FTS5 OR-expansion. Short English function
    # words that carry no retrieval signal and force false-negative AND
    # matches when left in the query.
    _FTS_STOPWORDS = frozenset({
        "a", "about", "above", "after", "again", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being",
        "between", "both", "but", "by", "can", "could", "did", "do", "does",
        "doing", "don", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "is", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "would",
        "you", "your", "yours", "yourself", "yourselves",
    })

    @classmethod
    def _sanitize_fts_query(cls, query: str) -> str:
        """Convert a natural-language query to an FTS5-safe OR expression.

        FTS5 treats a multi-word MATCH argument as AND-joined by default,
        which tanks recall on prose queries. This helper:
          - tokenizes the query
          - drops stopwords and short (<2 char) tokens
          - strips FTS5 special characters from each token
          - OR-joins the survivors

        If nothing remains (pathological query), falls back to the raw
        query so the caller sees zero results instead of a SQL error.
        """
        if not query:
            return ""
        # Strip FTS5 operator characters from EACH token to avoid
        # accidentally creating a malformed query.
        _FTS_SPECIAL = '"()*^:-+'
        tokens: list[str] = []
        for raw in query.lower().split():
            cleaned = raw.strip(".,;:!?\"'()[]{}#@<>") .translate(
                str.maketrans("", "", _FTS_SPECIAL)
            )
            if len(cleaned) < 2:
                continue
            if cleaned in cls._FTS_STOPWORDS:
                continue
            # FTS5 phrase-literal each token to ensure no special chars
            # sneak through as operators.
            tokens.append(f'"{cleaned}"')
        if not tokens:
            # Fallback: raw query (likely returns 0, but never crashes)
            return query
        return " OR ".join(tokens)

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
