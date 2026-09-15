from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from rag_graph_demo.domain.tkg import (
    AssertionMention,
    AssertionRecord,
    ConflictCase,
    EntityRecord,
    EventEffect,
    TKGExpansionRequest,
    TKGExpansionResponse,
    TKGQueryRequest,
    TKGQueryResponse,
    TKGSnapshot,
)

logger = logging.getLogger(__name__)


def _props(model: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    value = model.model_dump(mode="json", exclude=exclude or set())
    return {
        key: item
        for key, item in value.items()
        if item is not None and not isinstance(item, (dict, list))
    }


class Neo4jTKGStore:
    """Neo4j instance store for the full statement TKG.

    Every project node carries the :RagTKG label. Snapshot replacement deletes only
    those nodes, never unrelated data in the same Neo4j database.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
        workspace_id: str = "default",
        driver: AsyncDriver | None = None,
        write_batch_size: int = 500,
    ) -> None:
        self._driver = driver or AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._workspace_id = workspace_id
        self._write_batch_size = max(1, write_batch_size)

    async def close(self) -> None:
        await self._driver.close()

    async def verify(self) -> dict[str, Any]:
        await self._driver.verify_connectivity()
        async with self._driver.session(database=self._database) as session:
            record = await session.run(
                "MATCH (n:RagTKG {workspace_id: $workspace_id}) RETURN count(n) AS nodes",
                workspace_id=self._workspace_id,
            )
            row = await record.single()
        return {
            "ok": True,
            "database": self._database,
            "workspace_id": self._workspace_id,
            "nodes": row["nodes"],
        }

    async def initialize_schema(self) -> None:
        legacy_constraints = (
            "tkg_entity_id",
            "tkg_assertion_id",
            "tkg_event_id",
            "tkg_mention_id",
            "tkg_effect_id",
            "tkg_conflict_id",
            "tkg_decision_id",
            "tkg_review_id",
            "tkg_merge_id",
        )
        constraints = (
            ("entity", "RagTKGEntity", "entity_id"),
            ("assertion", "RagTKGAssertion", "assertion_id"),
            ("event", "RagTKGEvent", "event_id"),
            ("predicate", "RagTKGPredicate", "predicate_id"),
            ("document", "RagTKGDocument", "document_id"),
            ("mention", "RagTKGMention", "mention_id"),
            ("effect", "RagTKGEffect", "effect_id"),
            ("conflict", "RagTKGConflict", "conflict_id"),
        )
        indexes = (
            "CREATE FULLTEXT INDEX tkg_entity_names IF NOT EXISTS "
            "FOR (n:RagTKGEntity) ON EACH [n.canonical_name, n.alias_text]",
        )
        async with self._driver.session(database=self._database) as session:
            # Audit records may intentionally repeat after resumable snapshot imports.
            # Do not force uniqueness on them; provenance must never be deleted merely
            # to make a migration pass.
            for name in (
                "tkg_decision_workspace_id",
                "tkg_review_workspace_id",
                "tkg_merge_workspace_id",
            ):
                await session.run(f"DROP CONSTRAINT {name} IF EXISTS")
            # Existing installations predate workspaces. Assign them to the default
            # workspace before replacing global-ID uniqueness with scoped uniqueness.
            await session.run(
                "MATCH (n:RagTKG) WHERE n.workspace_id IS NULL "
                "SET n.workspace_id = 'default'"
            )
            for name in legacy_constraints:
                await session.run(f"DROP CONSTRAINT {name} IF EXISTS")
            for name, label, property_name in constraints:
                await session.run(
                    f"CREATE CONSTRAINT tkg_{name}_workspace_id IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE (n.workspace_id, n.{property_name}) IS UNIQUE"
                )
            for statement in indexes:
                await session.run(statement)
            # Create then remove schema tokens so optional properties/relationships
            # do not produce first-query warnings on an otherwise empty database.
            await session.run(
                """
                CREATE (a:RagTKG:RagTKGAssertion {
                  workspace_id: $workspace_id,
                  assertion_id: '__schema_probe__', valid_from: null, valid_to: null
                })
                CREATE (c:RagTKG:RagTKGConflict {
                  workspace_id: $workspace_id, conflict_id: '__schema_probe__'
                })
                CREATE (c)-[:INVOLVES]->(a)
                WITH a, c DETACH DELETE a, c
                """,
                workspace_id=self._workspace_id,
            )

    async def replace_snapshot(
        self,
        snapshot: TKGSnapshot,
        *,
        events: list[Any] | None = None,
        event_mentions: list[Any] | None = None,
        event_relations: list[Any] | None = None,
    ) -> None:
        event_by_id = {item.event_id: item for item in events or []}
        payload = self._payload(
            snapshot,
            event_by_id,
            event_mentions or [],
            event_relations or [],
        )
        for rows in payload.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                target = row.get("props", row)
                target["workspace_id"] = self._workspace_id

        stages = (
            (
                "entities",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGEntity)
                SET n = row.props, n.aliases = row.aliases,
                    n.source_document_ids = row.source_document_ids,
                    n.alias_text = row.alias_text
                """,
            ),
            (
                "predicates",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGPredicate)
                SET n = row
                """,
            ),
            (
                "assertions",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGAssertion)
                SET n = row.props, n.mention_ids = row.mention_ids,
                    n.event_effect_ids = row.event_effect_ids
                WITH n, row
                MATCH (s:RagTKGEntity {
                  workspace_id: $workspace_id, entity_id: row.subject_id
                })
                MATCH (o:RagTKGEntity {
                  workspace_id: $workspace_id, entity_id: row.object_id
                })
                MATCH (p:RagTKGPredicate {
                  workspace_id: $workspace_id, predicate_id: n.predicate_id
                })
                CREATE (n)-[:SUBJECT]->(s)
                CREATE (n)-[:OBJECT]->(o)
                CREATE (n)-[:PREDICATE]->(p)
                """,
            ),
            (
                "events",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGEvent)
                SET n = row.props, n.actors = row.actors, n.objects = row.objects,
                    n.source_document_ids = row.source_document_ids
                """,
            ),
            (
                "event_relations",
                """
                UNWIND $rows AS row
                MATCH (source:RagTKGEvent {
                  workspace_id: $workspace_id, event_id: row.source_event_id
                })
                MATCH (target:RagTKGEvent {
                  workspace_id: $workspace_id, event_id: row.target_event_id
                })
                CREATE (source)-[:RELATED_EVENT {
                  relation_id: row.relation_id,
                  relation_type: row.relation_type,
                  confidence: row.confidence,
                  shared_entities: row.shared_entities
                }]->(target)
                """,
            ),
            (
                "documents",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGDocument)
                SET n = row
                """,
            ),
            (
                "mentions",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGMention)
                SET n = row.props
                WITH n, row
                OPTIONAL MATCH (e:RagTKGEntity {
                  workspace_id: $workspace_id, entity_id: row.entity_id
                })
                OPTIONAL MATCH (a:RagTKGAssertion {
                  workspace_id: $workspace_id, assertion_id: row.assertion_id
                })
                OPTIONAL MATCH (v:RagTKGEvent {
                  workspace_id: $workspace_id, event_id: row.event_id
                })
                OPTIONAL MATCH (d:RagTKGDocument {
                  workspace_id: $workspace_id, document_id: row.document_id
                })
                FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
                  CREATE (n)-[:MENTIONS]->(e))
                FOREACH (_ IN CASE WHEN a IS NULL THEN [] ELSE [1] END |
                  CREATE (n)-[:SUPPORTS]->(a))
                FOREACH (_ IN CASE WHEN v IS NULL THEN [] ELSE [1] END |
                  CREATE (n)-[:MENTIONS_EVENT]->(v))
                FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
                  CREATE (n)-[:FROM_DOCUMENT]->(d))
                """,
            ),
            (
                "event_links",
                """
                UNWIND $rows AS row
                MATCH (v:RagTKGEvent {
                  workspace_id: $workspace_id, event_id: row.event_id
                })
                MATCH (e:RagTKGEntity {
                  workspace_id: $workspace_id, entity_id: row.entity_id
                })
                CREATE (v)-[:HAS_PARTICIPANT {
                  link_id: row.link_id, role: row.role,
                  surface_form: row.surface_form, document_id: row.document_id
                }]->(e)
                """,
            ),
            (
                "effects",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGEffect)
                SET n = row.props
                WITH n, row
                MATCH (v:RagTKGEvent {
                  workspace_id: $workspace_id, event_id: row.event_id
                })
                MATCH (a:RagTKGAssertion {
                  workspace_id: $workspace_id, assertion_id: row.assertion_id
                })
                CREATE (v)-[:HAS_EFFECT]->(n)-[:AFFECTS]->(a)
                """,
            ),
            (
                "conflicts",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGConflict)
                SET n = row.props, n.assertion_ids = row.assertion_ids,
                    n.mention_ids = row.mention_ids
                WITH n, row
                UNWIND row.assertion_ids AS assertion_id
                MATCH (a:RagTKGAssertion {
                  workspace_id: $workspace_id, assertion_id: assertion_id
                })
                CREATE (n)-[:INVOLVES]->(a)
                """,
            ),
            (
                "decisions",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGDecision)
                SET n = row.props, n.candidate_entity_ids = row.candidate_entity_ids
                WITH n, row
                OPTIONAL MATCH (e:RagTKGEntity {
                  workspace_id: $workspace_id, entity_id: row.selected_entity_id
                })
                FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
                  CREATE (n)-[:SELECTED]->(e))
                """,
            ),
            (
                "reviews",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGReview)
                SET n = row.props, n.candidate_entity_ids = row.candidate_entity_ids,
                    n.candidate_scores_json = row.candidate_scores_json
                WITH n, row
                OPTIONAL MATCH (e:RagTKGEntity {
                  workspace_id: $workspace_id, entity_id: row.provisional_entity_id
                })
                FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
                  CREATE (n)-[:PROVISIONAL]->(e))
                """,
            ),
            (
                "merges",
                """
                UNWIND $rows AS row
                CREATE (n:RagTKG:RagTKGMerge)
                SET n = row.props
                WITH n, row
                OPTIONAL MATCH (s:RagTKGEntity {
                  workspace_id: $workspace_id, entity_id: row.source_entity_id
                })
                OPTIONAL MATCH (t:RagTKGEntity {
                  workspace_id: $workspace_id, entity_id: row.target_entity_id
                })
                FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
                  CREATE (n)-[:SOURCE]->(s))
                FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END |
                  CREATE (n)-[:TARGET]->(t))
                """,
            ),
        )
        async with self._driver.session(database=self._database) as session:
            deleted = 0
            while True:
                count = await session.execute_write(
                    self._delete_batch_tx,
                    self._workspace_id,
                    self._write_batch_size,
                )
                deleted += count
                if count == 0:
                    break
            if deleted:
                logger.info(
                    "Deleted %s existing Neo4j nodes for workspace %s",
                    deleted,
                    self._workspace_id,
                )
            for name, statement in stages:
                rows = payload[name]
                if not rows:
                    continue
                for start in range(0, len(rows), self._write_batch_size):
                    await session.execute_write(
                        self._write_batch_tx,
                        statement,
                        rows[start : start + self._write_batch_size],
                        self._workspace_id,
                    )
                logger.info(
                    "Synced %s Neo4j %s row(s) for workspace %s",
                    len(rows),
                    name,
                    self._workspace_id,
                )

    @staticmethod
    async def _delete_batch_tx(
        tx: Any,
        workspace_id: str,
        batch_size: int,
    ) -> int:
        result = await tx.run(
            """
            MATCH (n:RagTKG {workspace_id: $workspace_id})
            WITH n LIMIT $batch_size
            DETACH DELETE n
            RETURN count(n) AS deleted
            """,
            workspace_id=workspace_id,
            batch_size=batch_size,
        )
        row = await result.single()
        return int(row["deleted"] if row else 0)

    @staticmethod
    async def _write_batch_tx(
        tx: Any,
        statement: str,
        rows: list[dict[str, Any]],
        workspace_id: str,
    ) -> None:
        result = await tx.run(
            statement,
            rows=rows,
            workspace_id=workspace_id,
        )
        await result.consume()

    async def query(self, request: TKGQueryRequest) -> TKGQueryResponse:
        conditions = [
            "($entity_id IS NULL OR s.entity_id = $entity_id OR o.entity_id = $entity_id)",
            "($predicate_id IS NULL OR a.predicate_id = $predicate_id)",
            "($include_inactive OR NOT a.status IN ['inactive','superseded','retracted'])",
            "($include_disputed OR a.status <> 'disputed')",
            "($at_time IS NULL OR a.valid_from = '' OR a.valid_from <= $at_time)",
            "($at_time IS NULL OR a.valid_to = '' OR a.valid_to >= $at_time)",
        ]
        if request.entity_name:
            conditions.append(
                "(toLower(s.canonical_name) = toLower($entity_name) "
                "OR toLower(o.canonical_name) = toLower($entity_name) "
                "OR any(alias IN coalesce(s.aliases, []) "
                "WHERE toLower(alias) = toLower($entity_name)) "
                "OR any(alias IN coalesce(o.aliases, []) "
                "WHERE toLower(alias) = toLower($entity_name)))"
            )
        query = f"""
        MATCH (a:RagTKGAssertion {{workspace_id: $workspace_id}})-[:SUBJECT]->(s:RagTKGEntity)
        MATCH (a)-[:OBJECT]->(o:RagTKGEntity)
        WHERE {' AND '.join(conditions)}
        WITH a, s, o ORDER BY coalesce(a.valid_from, '') DESC LIMIT $limit
        OPTIONAL MATCH (m:RagTKGMention)-[:SUPPORTS]->(a)
        OPTIONAL MATCH (v:RagTKGEvent)-[:HAS_EFFECT]->(ef:RagTKGEffect)-[:AFFECTS]->(a)
        OPTIONAL MATCH (c:RagTKGConflict)-[:INVOLVES]->(a)
        RETURN collect(DISTINCT properties(a)) AS assertions,
               collect(DISTINCT properties(s)) + collect(DISTINCT properties(o)) AS entities,
               collect(DISTINCT properties(m)) AS mentions,
               collect(DISTINCT properties(ef)) AS effects,
               collect(DISTINCT properties(c)) AS conflicts
        """
        parameters = {
            **request.model_dump(mode="json"),
            "workspace_id": self._workspace_id,
        }
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, **parameters)
            row = await result.single()
        if row is None:
            return TKGQueryResponse()
        entities = {
            item["entity_id"]: EntityRecord.model_validate(item)
            for item in row["entities"] or []
            if item and item.get("entity_id")
        }
        assertions = [
            AssertionRecord.model_validate(item)
            for item in row["assertions"] or []
            if item and item.get("assertion_id")
        ]
        mentions = [
            AssertionMention.model_validate(item)
            for item in row["mentions"] or []
            if item and item.get("assertion_id")
        ]
        effects = [
            EventEffect.model_validate(item)
            for item in row["effects"] or []
            if item and item.get("effect_id")
        ]
        conflicts = [
            ConflictCase.model_validate(item)
            for item in row["conflicts"] or []
            if item and item.get("conflict_id")
        ]
        return TKGQueryResponse(
            entities=list(entities.values()),
            assertions=assertions,
            mentions=mentions,
            effects=effects,
            conflicts=conflicts,
        )

    async def query_graph(
        self,
        *,
        entity_name: str | None = None,
        limit: int = 200,
    ) -> dict[str, list[dict[str, Any]]]:
        where = ""
        parameters: dict[str, Any] = {
            "limit": limit,
            "workspace_id": self._workspace_id,
        }
        if entity_name:
            where = (
                "WHERE toLower(e.canonical_name) CONTAINS toLower($entity_name) "
                "OR any(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($entity_name))"
            )
            parameters["entity_name"] = entity_name
        query = f"""
        MATCH (e:RagTKGEntity {{workspace_id: $workspace_id}})
        {where}
        OPTIONAL MATCH (a:RagTKGAssertion)-[:SUBJECT|OBJECT]->(e)
        OPTIONAL MATCH (a)-[:SUBJECT]->(s:RagTKGEntity)
        OPTIONAL MATCH (a)-[:OBJECT]->(o:RagTKGEntity)
        RETURN collect(DISTINCT e {{.*}})[..$limit] AS seed_entities,
               collect(DISTINCT a {{.*}})[..$limit] AS assertions,
               collect(DISTINCT s {{.*}})[..$limit] +
               collect(DISTINCT o {{.*}})[..$limit] AS linked_entities
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, **parameters)
            row = await result.single()
        entities = {
            item["entity_id"]: item
            for item in [*(row["seed_entities"] or []), *(row["linked_entities"] or [])]
            if item and item.get("entity_id")
        }
        assertions = [item for item in row["assertions"] or [] if item]
        nodes = [
            {"id": key, "type": "entity", "label": value.get("canonical_name"), **value}
            for key, value in entities.items()
        ]
        edges = []
        for assertion in assertions:
            nodes.append(
                {
                    "id": assertion["assertion_id"],
                    "type": "assertion",
                    "label": assertion.get("predicate_id"),
                    **assertion,
                }
            )
            edges.extend(
                [
                    {
                        "source": assertion["subject_id"],
                        "target": assertion["assertion_id"],
                        "type": "SUBJECT",
                    },
                    {
                        "source": assertion["assertion_id"],
                        "target": assertion["object_id"],
                        "type": "OBJECT",
                    },
                ]
            )
        return {"nodes": nodes[:limit], "edges": edges[: limit * 2]}

    async def expand(self, request: TKGExpansionRequest) -> TKGExpansionResponse:
        """Expand vector seeds through approved TKG paths and return source evidence."""
        parameters = {
            **request.model_dump(mode="json"),
            "workspace_id": self._workspace_id,
        }
        query = """
        MATCH (a:RagTKGAssertion {workspace_id: $workspace_id})
        WHERE (
          a.assertion_id IN $assertion_ids OR a.predicate_id IN $predicate_ids OR
          EXISTS {
            MATCH (a)-[:SUBJECT|OBJECT]->(seed:RagTKGEntity)
            WHERE seed.entity_id IN $entity_ids
          } OR EXISTS {
            MATCH (seed_event:RagTKGEvent)-[:HAS_EFFECT]->(:RagTKGEffect)-[:AFFECTS]->(a)
            WHERE seed_event.workspace_id = $workspace_id
              AND seed_event.event_id IN $event_ids
              AND $include_event_effects
          }
        )
        AND ($include_inactive OR NOT a.status IN ['inactive','superseded','retracted'])
        AND ($include_disputed OR a.status <> 'disputed')
        AND ($start_time IS NULL OR a.valid_to = '' OR a.valid_to >= $start_time)
        AND ($end_time IS NULL OR a.valid_from = '' OR a.valid_from <= $end_time)
        MATCH (a)-[:SUBJECT]->(s:RagTKGEntity)
        MATCH (a)-[:OBJECT]->(o:RagTKGEntity)
        WITH DISTINCT a, s, o,
             coalesce($assertion_scores[a.assertion_id], 0.0) * 0.45 +
             coalesce($predicate_scores[a.predicate_id], 0.0) * 0.20 +
             CASE WHEN coalesce($entity_scores[s.entity_id], 0.0) >=
                            coalesce($entity_scores[o.entity_id], 0.0)
                  THEN coalesce($entity_scores[s.entity_id], 0.0)
                  ELSE coalesce($entity_scores[o.entity_id], 0.0) END * 0.20 +
             CASE WHEN coalesce($entity_scores[s.entity_id], 0.0) > 0 AND
                            coalesce($entity_scores[o.entity_id], 0.0) > 0
                  THEN 0.15 ELSE 0 END AS seed_score
        ORDER BY CASE WHEN $rank_by_seed_score THEN seed_score ELSE 0.0 END DESC
        LIMIT $limit
        OPTIONAL MATCH (m:RagTKGMention)-[:SUPPORTS]->(a)
        OPTIONAL MATCH (v:RagTKGEvent)-[:HAS_EFFECT]->(ef:RagTKGEffect)-[:AFFECTS]->(a)
        WHERE $include_event_effects
        OPTIONAL MATCH (c:RagTKGConflict)-[:INVOLVES]->(a)
        RETURN collect(DISTINCT properties(a)) AS assertions,
               collect(DISTINCT properties(s)) + collect(DISTINCT properties(o)) AS entities,
               collect(DISTINCT properties(m)) AS mentions,
               collect(DISTINCT properties(v)) AS events,
               collect(DISTINCT properties(ef)) AS effects,
               collect(DISTINCT properties(c)) AS conflicts
        """
        event_query = """
        MATCH (v:RagTKGEvent {workspace_id: $workspace_id})
        WHERE v.event_id IN $event_ids OR EXISTS {
          MATCH (v)-[:HAS_PARTICIPANT]->(e:RagTKGEntity)
          WHERE e.entity_id IN $entity_ids
        }
        WITH DISTINCT v
        OPTIONAL MATCH (v)-[r:RELATED_EVENT]-(linked:RagTKGEvent)
        WITH collect(DISTINCT v) + CASE WHEN $expand_event_chain
             THEN collect(DISTINCT linked) ELSE [] END AS values
        UNWIND values AS event
        WITH DISTINCT event WHERE event IS NOT NULL
        WITH event, coalesce($event_scores[event.event_id], 0.0) AS seed_score
        ORDER BY CASE WHEN $rank_by_seed_score THEN seed_score ELSE 0.0 END DESC
        LIMIT $limit
        OPTIONAL MATCH (em:RagTKGMention)-[:MENTIONS_EVENT]->(event)
        OPTIONAL MATCH (event)-[participant:HAS_PARTICIPANT]->(entity:RagTKGEntity)
        RETURN collect(DISTINCT properties(event)) AS events,
               collect(DISTINCT properties(em)) AS event_mentions,
               collect(DISTINCT CASE WHEN participant IS NULL THEN null ELSE {
                 event_id: event.event_id,
                 entity_id: entity.entity_id,
                 role: participant.role,
                 surface_form: participant.surface_form,
                 document_id: participant.document_id
               } END) AS event_links
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, **parameters)
            row = await result.single()
            event_result = await session.run(event_query, **parameters)
            event_row = await event_result.single()
        row = row or {}
        event_row = event_row or {}
        entities = {
            item["entity_id"]: EntityRecord.model_validate(item)
            for item in row.get("entities", []) or []
            if item and item.get("entity_id")
        }
        assertions = [
            AssertionRecord.model_validate(item)
            for item in row.get("assertions", []) or []
            if item and item.get("assertion_id")
        ]
        mentions = [
            AssertionMention.model_validate(item)
            for item in row.get("mentions", []) or []
            if item and item.get("assertion_id")
        ]
        effects = [
            EventEffect.model_validate(item)
            for item in row.get("effects", []) or []
            if item and item.get("effect_id")
        ]
        conflicts = [
            ConflictCase.model_validate(item)
            for item in row.get("conflicts", []) or []
            if item and item.get("conflict_id")
        ]
        events = {
            item["event_id"]: item
            for item in [
                *(row.get("events", []) or []),
                *(event_row.get("events", []) or []),
            ]
            if item and item.get("event_id")
        }
        event_mentions = {
            item["mention_id"]: item
            for item in event_row.get("event_mentions", []) or []
            if item and item.get("mention_id")
        }
        return TKGExpansionResponse(
            entities=list(entities.values()),
            assertions=assertions,
            mentions=mentions,
            events=list(events.values()),
            event_mentions=list(event_mentions.values()),
            event_links=[
                item
                for item in event_row.get("event_links", []) or []
                if item and item.get("event_id") and item.get("entity_id")
            ],
            effects=effects,
            conflicts=conflicts,
        )

    @staticmethod
    def _payload(
        snapshot: TKGSnapshot,
        events: dict[str, Any],
        event_mentions: list[Any],
        event_relations: list[Any],
    ) -> dict[str, Any]:
        import json

        mention_documents = {
            item.document_id: getattr(item, "document_title", "")
            for item in [
                *snapshot.entity_mentions,
                *snapshot.assertion_mentions,
                *event_mentions,
            ]
        }
        return {
            "documents": [
                {"document_id": key, "title": value}
                for key, value in mention_documents.items()
            ],
            "entities": [
                {
                    "props": _props(
                        item,
                        exclude={"aliases", "mention_ids", "source_document_ids"},
                    ),
                    "aliases": item.aliases,
                    "source_document_ids": item.source_document_ids,
                    "alias_text": " ".join(item.aliases),
                }
                for item in snapshot.entities
            ],
            "assertions": [
                {
                    "props": {
                        **_props(item, exclude={"mention_ids", "event_effect_ids"}),
                        "valid_from": item.valid_from or "",
                        "valid_to": item.valid_to or "",
                        "stated_valid_to": item.stated_valid_to or "",
                    },
                    "subject_id": item.subject_id,
                    "object_id": item.object_id,
                    "mention_ids": item.mention_ids,
                    "event_effect_ids": item.event_effect_ids,
                }
                for item in snapshot.assertions
            ],
            "predicates": [
                {
                    "workspace_id": "",
                    "predicate_id": predicate_id,
                    "canonical_name": predicate_id,
                    "description": "",
                    "ontology_version": snapshot.ontology_version,
                }
                for predicate_id in sorted(
                    {item.predicate_id for item in snapshot.assertions}
                )
            ],
            "events": [
                {
                    "props": _props(
                        item,
                        exclude={
                            "actors",
                            "objects",
                            "source_document_ids",
                            "mention_ids",
                            "fact_effects",
                        },
                    ),
                    "actors": item.actors,
                    "objects": item.objects,
                    "source_document_ids": item.source_document_ids,
                }
                for item in events.values()
            ],
            "mentions": [
                {
                    "props": {**_props(item), "mention_kind": "entity"},
                    "entity_id": item.entity_id,
                    "assertion_id": None,
                    "event_id": None,
                    "document_id": item.document_id,
                }
                for item in snapshot.entity_mentions
            ]
            + [
                {
                    "props": {**_props(item), "mention_kind": "assertion"},
                    "entity_id": None,
                    "assertion_id": item.assertion_id,
                    "event_id": None,
                    "document_id": item.document_id,
                }
                for item in snapshot.assertion_mentions
            ]
            + [
                {
                    "props": {**_props(item), "mention_kind": "event"},
                    "entity_id": None,
                    "assertion_id": None,
                    "event_id": item.event_id,
                    "document_id": item.document_id,
                }
                for item in event_mentions
                if item.event_id in events
            ],
            "event_links": [item.model_dump(mode="json") for item in snapshot.event_entity_links],
            "event_relations": [
                item.model_dump(mode="json")
                for item in event_relations
                if item.source_event_id in events and item.target_event_id in events
            ],
            "effects": [
                {
                    "props": _props(item),
                    "event_id": item.event_id,
                    "assertion_id": item.assertion_id,
                }
                for item in snapshot.event_effects
            ],
            "conflicts": [
                {
                    "props": _props(item, exclude={"assertion_ids", "mention_ids"}),
                    "assertion_ids": item.assertion_ids,
                    "mention_ids": item.mention_ids,
                }
                for item in snapshot.conflicts
            ],
            "decisions": [
                {
                    "props": _props(item, exclude={"candidate_entity_ids"}),
                    "selected_entity_id": item.selected_entity_id,
                    "candidate_entity_ids": item.candidate_entity_ids,
                }
                for item in snapshot.resolution_decisions
            ],
            "reviews": [
                {
                    "props": _props(item, exclude={"candidates"}),
                    "provisional_entity_id": item.provisional_entity_id,
                    "candidate_entity_ids": [value.entity_id for value in item.candidates],
                    "candidate_scores_json": json.dumps(
                        [value.model_dump(mode="json") for value in item.candidates],
                        ensure_ascii=False,
                    ),
                }
                for item in snapshot.review_cases
            ],
            "merges": [
                {
                    "props": _props(
                        item,
                        exclude={
                            "source_entity",
                            "target_entity_before",
                            "moved_entity_mention_ids",
                            "moved_assertion_subject_ids",
                            "moved_assertion_object_ids",
                            "moved_event_link_ids",
                            "assertion_coalescences",
                            "conflict_snapshots",
                        },
                    ),
                    "source_entity_id": item.source_entity.entity_id,
                    "target_entity_id": item.target_entity_id,
                }
                for item in snapshot.merge_records
            ],
        }
