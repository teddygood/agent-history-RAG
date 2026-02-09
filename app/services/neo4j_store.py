from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from neo4j import GraphDatabase

from app.core.config import settings


class Neo4jStore:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def init_schema(self) -> None:
        queries = [
            "CREATE CONSTRAINT turn_uid_unique IF NOT EXISTS FOR (t:Turn) REQUIRE t.turn_uid IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE FULLTEXT INDEX turn_text_fulltext IF NOT EXISTS FOR (t:Turn) ON EACH [t.text]",
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    def upsert_turn(
        self,
        *,
        turn_uid: str,
        conversation_id: str,
        turn_id: str,
        speaker: str,
        text: str,
        timestamp: datetime,
        embedding: list[float],
        importance_score: float = 0.0,
        chunk_profile: str = "default",
        chunk_count: int = 1,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        query = """
        MERGE (t:Turn {turn_uid: $turn_uid})
        SET t.conversation_id = $conversation_id,
            t.turn_id = $turn_id,
            t.speaker = $speaker,
            t.text = $text,
            t.timestamp = datetime($timestamp),
            t.embedding = $embedding,
            t.importance_score = $importance_score,
            t.chunk_profile = $chunk_profile,
            t.chunk_count = $chunk_count,
            t.chunk_size = $chunk_size,
            t.chunk_overlap = $chunk_overlap,
            t.recall_count = coalesce(t.recall_count, 0),
            t.updated_at = datetime()
        """
        params = {
            "turn_uid": turn_uid,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "speaker": speaker,
            "text": text,
            "timestamp": timestamp.isoformat(),
            "embedding": embedding,
            "importance_score": importance_score,
            "chunk_profile": chunk_profile,
            "chunk_count": chunk_count,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        with self.driver.session() as session:
            session.run(query, params)

    def upsert_turns_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        query = """
        UNWIND $rows AS row
        MERGE (t:Turn {turn_uid: row.turn_uid})
        SET t.conversation_id = row.conversation_id,
            t.turn_id = row.turn_id,
            t.speaker = row.speaker,
            t.text = row.text,
            t.timestamp = datetime(row.timestamp),
            t.embedding = row.embedding,
            t.importance_score = row.importance_score,
            t.chunk_profile = row.chunk_profile,
            t.chunk_count = row.chunk_count,
            t.chunk_size = row.chunk_size,
            t.chunk_overlap = row.chunk_overlap,
            t.recall_count = coalesce(t.recall_count, 0),
            t.updated_at = datetime()
        """

        payload: list[dict[str, Any]] = []
        for row in rows:
            copied = dict(row)
            timestamp = copied.get("timestamp")
            if hasattr(timestamp, "isoformat"):
                copied["timestamp"] = timestamp.isoformat()
            payload.append(copied)

        with self.driver.session() as session:
            session.run(query, {"rows": payload})

    def upsert_entity(
        self,
        *,
        entity_id: str,
        canonical_name: str,
        alias: str,
        entity_type: str,
        description: str,
        embedding: list[float],
    ) -> None:
        query = """
        MERGE (e:Entity {entity_id: $entity_id})
        ON CREATE SET e.canonical_name = $canonical_name,
                      e.entity_type = $entity_type,
                      e.description = $description,
                      e.embedding = $embedding,
                      e.aliases = [$alias],
                      e.created_at = datetime()
        ON MATCH SET e.entity_type = coalesce(e.entity_type, $entity_type),
                     e.description = CASE WHEN e.description = '' THEN $description ELSE e.description END,
                     e.embedding = $embedding,
                     e.aliases = CASE
                       WHEN e.aliases IS NULL THEN [$alias]
                       WHEN NOT $alias IN e.aliases THEN e.aliases + $alias
                       ELSE e.aliases
                     END,
                     e.updated_at = datetime()
        """
        params = {
            "entity_id": entity_id,
            "canonical_name": canonical_name,
            "alias": alias,
            "entity_type": entity_type,
            "description": description,
            "embedding": embedding,
        }
        with self.driver.session() as session:
            session.run(query, params)

    def upsert_entities_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        query = """
        UNWIND $rows AS row
        MERGE (e:Entity {entity_id: row.entity_id})
        ON CREATE SET e.canonical_name = row.canonical_name,
                      e.entity_type = row.entity_type,
                      e.description = row.description,
                      e.embedding = row.embedding,
                      e.aliases = row.aliases,
                      e.created_at = datetime()
        ON MATCH SET e.entity_type = coalesce(e.entity_type, row.entity_type),
                     e.description = CASE WHEN e.description = '' THEN row.description ELSE e.description END,
                     e.aliases = reduce(
                        acc = coalesce(e.aliases, []),
                        alias IN coalesce(row.aliases, []) |
                        CASE WHEN alias IN acc THEN acc ELSE acc + alias END
                     ),
                     e.updated_at = datetime()
        """

        with self.driver.session() as session:
            session.run(query, {"rows": rows})

    def link_turn_entity(self, *, turn_uid: str, entity_id: str) -> None:
        query = """
        MATCH (t:Turn {turn_uid: $turn_uid})
        MATCH (e:Entity {entity_id: $entity_id})
        MERGE (t)-[m:MENTIONS]->(e)
        SET m.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(query, {"turn_uid": turn_uid, "entity_id": entity_id})

    def link_turn_entities_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        query = """
        UNWIND $rows AS row
        MATCH (t:Turn {turn_uid: row.turn_uid})
        MATCH (e:Entity {entity_id: row.entity_id})
        MERGE (t)-[m:MENTIONS]->(e)
        SET m.updated_at = datetime()
        """

        with self.driver.session() as session:
            session.run(query, {"rows": rows})

    def upsert_relation(
        self,
        *,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        evidence_turn_uid: str,
        weight_increment: float = 1.0,
    ) -> None:
        query = """
        MATCH (s:Entity {entity_id: $source_entity_id})
        MATCH (t:Entity {entity_id: $target_entity_id})
        MERGE (s)-[r:RELATES_TO {relation_type: $relation_type}]->(t)
        ON CREATE SET r.weight = $weight_increment,
                      r.evidence_turn_ids = [$evidence_turn_uid],
                      r.created_at = datetime()
        ON MATCH SET r.weight = coalesce(r.weight, 0.0) + $weight_increment,
                     r.evidence_turn_ids = CASE
                        WHEN r.evidence_turn_ids IS NULL THEN [$evidence_turn_uid]
                        WHEN NOT $evidence_turn_uid IN r.evidence_turn_ids THEN r.evidence_turn_ids + $evidence_turn_uid
                        ELSE r.evidence_turn_ids
                     END,
                     r.updated_at = datetime()
        """
        params = {
            "source_entity_id": source_entity_id,
            "target_entity_id": target_entity_id,
            "relation_type": relation_type,
            "evidence_turn_uid": evidence_turn_uid,
            "weight_increment": weight_increment,
        }
        with self.driver.session() as session:
            session.run(query, params)

    def upsert_relations_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        query = """
        UNWIND $rows AS row
        MATCH (s:Entity {entity_id: row.source_entity_id})
        MATCH (t:Entity {entity_id: row.target_entity_id})
        MERGE (s)-[r:RELATES_TO {relation_type: row.relation_type}]->(t)
        ON CREATE SET r.weight = row.weight_increment,
                      r.evidence_turn_ids = row.evidence_turn_ids,
                      r.created_at = datetime()
        ON MATCH SET r.weight = coalesce(r.weight, 0.0) + row.weight_increment,
                     r.evidence_turn_ids = reduce(
                        acc = coalesce(r.evidence_turn_ids, []),
                        ev IN coalesce(row.evidence_turn_ids, []) |
                        CASE WHEN ev IN acc THEN acc ELSE acc + ev END
                     ),
                     r.updated_at = datetime()
        """

        with self.driver.session() as session:
            session.run(query, {"rows": rows})

    def get_existing_turn_uids(self, turn_uids: Iterable[str]) -> set[str]:
        ids = list(dict.fromkeys(turn_uids))
        if not ids:
            return set()

        query = """
        MATCH (t:Turn)
        WHERE t.turn_uid IN $turn_uids
        RETURN t.turn_uid AS turn_uid
        """
        with self.driver.session() as session:
            records = session.run(query, {"turn_uids": ids})
            return {str(record["turn_uid"]) for record in records}

    def search_entities(self, query_text: str, limit: int = 25) -> list[dict[str, Any]]:
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.canonical_name) CONTAINS toLower($query_text)
           OR any(alias IN coalesce(e.aliases, []) WHERE toLower(alias) CONTAINS toLower($query_text))
        RETURN e.entity_id AS entity_id,
               e.canonical_name AS canonical_name,
               coalesce(e.aliases, []) AS aliases,
               coalesce(e.entity_type, 'concept') AS entity_type,
               coalesce(e.description, '') AS description,
               coalesce(e.embedding, []) AS embedding
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = session.run(query, {"query_text": query_text, "limit": limit})
            return [dict(record) for record in records]

    def get_neighbors(self, entity_id: str, limit: int = 100) -> list[dict[str, Any]]:
        query = """
        MATCH (s:Entity {entity_id: $entity_id})-[r:RELATES_TO]-(n:Entity)
        RETURN s.entity_id AS source_entity_id,
               s.canonical_name AS source_name,
               n.entity_id AS target_entity_id,
               n.canonical_name AS target_name,
               coalesce(n.embedding, []) AS target_embedding,
               r.relation_type AS relation_type,
               coalesce(r.weight, 1.0) AS relation_weight,
               coalesce(r.evidence_turn_ids, []) AS evidence_turn_ids
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = session.run(query, {"entity_id": entity_id, "limit": limit})
            return [dict(record) for record in records]

    def fetch_turns_for_entities(
        self,
        entity_ids: Iterable[str],
        *,
        conversation_id: str | None = None,
        limit: int = 400,
    ) -> list[dict[str, Any]]:
        ids = list(entity_ids)
        if not ids:
            return []

        query = """
        MATCH (t:Turn)-[:MENTIONS]->(e:Entity)
        WHERE e.entity_id IN $entity_ids
          AND ($conversation_id IS NULL OR t.conversation_id = $conversation_id)
        RETURN t.turn_uid AS turn_uid,
               t.conversation_id AS conversation_id,
               t.turn_id AS turn_id,
               t.speaker AS speaker,
               t.text AS text,
               t.timestamp AS timestamp,
               coalesce(t.embedding, []) AS embedding,
               coalesce(t.importance_score, 0.0) AS importance_score,
               coalesce(t.chunk_profile, 'default') AS chunk_profile,
               coalesce(t.chunk_count, 1) AS chunk_count,
               coalesce(t.chunk_size, 0) AS chunk_size,
               coalesce(t.chunk_overlap, 0) AS chunk_overlap,
               t.last_recalled_at AS last_recalled_at,
               coalesce(t.recall_count, 0) AS recall_count,
               collect(DISTINCT e.entity_id) AS matched_entity_ids,
               collect(DISTINCT e.canonical_name) AS matched_entities
        LIMIT $limit
        """
        params = {
            "entity_ids": ids,
            "conversation_id": conversation_id,
            "limit": limit,
        }
        with self.driver.session() as session:
            records = session.run(query, params)
            return [dict(record) for record in records]

    def search_turns_fulltext(
        self,
        *,
        query_text: str,
        conversation_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if not query_text.strip():
            return []

        query = """
        CALL db.index.fulltext.queryNodes('turn_text_fulltext', $query_text, {limit: $limit})
        YIELD node, score
        WHERE ($conversation_id IS NULL OR node.conversation_id = $conversation_id)
        OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
        RETURN node.turn_uid AS turn_uid,
               node.conversation_id AS conversation_id,
               node.turn_id AS turn_id,
               node.speaker AS speaker,
               node.text AS text,
               node.timestamp AS timestamp,
               coalesce(node.embedding, []) AS embedding,
               coalesce(node.importance_score, 0.0) AS importance_score,
               coalesce(node.chunk_profile, 'default') AS chunk_profile,
               coalesce(node.chunk_count, 1) AS chunk_count,
               coalesce(node.chunk_size, 0) AS chunk_size,
               coalesce(node.chunk_overlap, 0) AS chunk_overlap,
               node.last_recalled_at AS last_recalled_at,
               coalesce(node.recall_count, 0) AS recall_count,
               coalesce(score, 0.0) AS lexical_score,
               collect(DISTINCT e.entity_id) AS matched_entity_ids,
               collect(DISTINCT e.canonical_name) AS matched_entities
        ORDER BY lexical_score DESC
        LIMIT $limit
        """
        params = {
            "query_text": query_text,
            "conversation_id": conversation_id,
            "limit": limit,
        }
        with self.driver.session() as session:
            records = session.run(query, params)
            return [dict(record) for record in records]

    def get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        query = """
        MATCH (e:Entity {entity_id: $entity_id})
        RETURN e.entity_id AS entity_id,
               e.canonical_name AS canonical_name,
               coalesce(e.aliases, []) AS aliases,
               coalesce(e.entity_type, 'concept') AS entity_type,
               coalesce(e.description, '') AS description
        """
        with self.driver.session() as session:
            record = session.run(query, {"entity_id": entity_id}).single()
            return dict(record) if record else None

    def get_turn_by_uid(self, turn_uid: str) -> dict[str, Any] | None:
        query = """
        MATCH (t:Turn {turn_uid: $turn_uid})
        RETURN t.turn_uid AS turn_uid,
               t.conversation_id AS conversation_id,
               t.turn_id AS turn_id,
               t.speaker AS speaker,
               t.text AS text,
               t.timestamp AS timestamp
        """
        with self.driver.session() as session:
            record = session.run(query, {"turn_uid": turn_uid}).single()
            return dict(record) if record else None

    def get_turns_by_conversation(self, conversation_id: str, limit: int = 10000) -> list[dict[str, Any]]:
        query = """
        MATCH (t:Turn {conversation_id: $conversation_id})
        RETURN t.turn_uid AS turn_uid,
               t.conversation_id AS conversation_id,
               t.turn_id AS turn_id,
               t.speaker AS speaker,
               t.text AS text,
               t.timestamp AS timestamp
        ORDER BY t.timestamp ASC
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = session.run(query, {"conversation_id": conversation_id, "limit": limit})
            return [dict(record) for record in records]

    def mark_turns_recalled(self, turn_uids: Iterable[str], recalled_at: datetime) -> None:
        ids = list(dict.fromkeys(turn_uids))
        if not ids:
            return

        query = """
        UNWIND $turn_uids AS turn_uid
        MATCH (t:Turn {turn_uid: turn_uid})
        SET t.last_recalled_at = datetime($recalled_at),
            t.recall_count = coalesce(t.recall_count, 0) + 1,
            t.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(query, {"turn_uids": ids, "recalled_at": recalled_at.isoformat()})

    def build_subgraph(self, seed: str, limit: int = 120) -> dict[str, list[dict[str, Any]]]:
        seeds = self.search_entities(seed, limit=8)
        node_map: dict[str, dict[str, Any]] = {}
        edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}

        for entity in seeds:
            node_map[entity["entity_id"]] = {
                "id": entity["entity_id"],
                "label": entity["canonical_name"],
                "kind": "entity",
                "score": 1.0,
            }

        for entity in seeds:
            for neighbor in self.get_neighbors(entity["entity_id"], limit=max(12, limit // max(1, len(seeds)))):
                src = neighbor["source_entity_id"]
                dst = neighbor["target_entity_id"]
                node_map.setdefault(
                    dst,
                    {
                        "id": dst,
                        "label": neighbor["target_name"],
                        "kind": "entity",
                        "score": 0.6,
                    },
                )
                key = (src, dst, neighbor["relation_type"])
                edge_map[key] = {
                    "source": src,
                    "target": dst,
                    "label": neighbor["relation_type"],
                    "evidence_turn_ids": neighbor["evidence_turn_ids"],
                }
                if len(edge_map) >= limit:
                    break
            if len(edge_map) >= limit:
                break

        return {"nodes": list(node_map.values()), "edges": list(edge_map.values())}
