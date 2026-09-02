from __future__ import annotations

import importlib.util
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from types import ModuleType
from typing import Any

from rag_graph_demo.domain.tkg import (
    SemanticEntityCandidate,
    SemanticExtractionBatch,
    TKGSnapshot,
)
from rag_graph_demo.services.entity_resolution_service import (
    identity_signature,
    normalize_name,
)

SHARED_ENTITY_TYPES = ("Person", "Organization", "Location")


def dwie_shared_type(tags: list[str]) -> str | None:
    values = set(tags)
    if "type::person" in values:
        return "Person"
    if "type::organization" in values:
        return "Organization"
    if "type::location" in values:
        return "Location"
    return None


def normalize_predicted_type(value: str) -> str | None:
    normalized = normalize_name(value)
    if normalized in {"person", "people", "human"}:
        return "Person"
    if normalized in {
        "organization",
        "organisation",
        "company",
        "institution",
        "agency",
        "government",
        "team",
        "ngo",
        "igo",
    }:
        return "Organization"
    if normalized in {
        "location",
        "place",
        "geo",
        "geography",
        "country",
        "city",
        "region",
        "facility",
    }:
        return "Location"
    return None


def coarsen_gold(document: dict[str, Any]) -> dict[str, Any]:
    """Restrict DWIE to the three entity types shared with the current ontology."""
    concept_types = {
        int(concept["concept"]): dwie_shared_type(concept.get("tags", []))
        for concept in document.get("concepts", [])
    }
    retained_ids = [concept_id for concept_id, entity_type in concept_types.items() if entity_type]
    remap = {concept_id: index for index, concept_id in enumerate(retained_ids)}
    concepts = [
        {
            "concept": remap[concept_id],
            "text": next(
                (
                    concept.get("text", "")
                    for concept in document.get("concepts", [])
                    if int(concept["concept"]) == concept_id
                ),
                "",
            ),
            "tags": [concept_types[concept_id]],
        }
        for concept_id in retained_ids
    ]
    mentions = [
        {
            "begin": int(mention["begin"]),
            "end": int(mention["end"]),
            "text": mention.get("text", ""),
            "concept": remap[int(mention["concept"])],
            "tags": [concept_types[int(mention["concept"])]],
        }
        for mention in document.get("mentions", [])
        if int(mention.get("concept", -1)) in remap
    ]
    return {
        "id": document["id"],
        "content": document.get("content", ""),
        "tags": document.get("tags", []),
        "mentions": mentions,
        "concepts": concepts,
        "relations": [],
    }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _prediction_from_clusters(
    scoped: dict[str, Any], clusters: list[list[int]]
) -> dict[str, Any]:
    mentions = scoped["mentions"]
    concept_by_mention = {
        mention_index: concept_index
        for concept_index, cluster in enumerate(clusters)
        for mention_index in cluster
    }
    concepts = []
    for concept_index, cluster in enumerate(clusters):
        representative = max(cluster, key=lambda index: len(mentions[index]["text"]))
        concepts.append(
            {
                "concept": concept_index,
                "text": mentions[representative]["text"],
                "tags": mentions[representative]["tags"],
            }
        )
    return {
        **scoped,
        "concepts": concepts,
        "mentions": [
            {**mention, "concept": concept_by_mention[index]}
            for index, mention in enumerate(mentions)
        ],
    }


def singleton_prediction(gold: dict[str, Any]) -> dict[str, Any]:
    """Treat every Gold mention as a separate entity."""
    scoped = coarsen_gold(gold)
    return _prediction_from_clusters(
        scoped, [[index] for index in range(len(scoped["mentions"]))]
    )


def exact_match_prediction(gold: dict[str, Any]) -> dict[str, Any]:
    """Cluster Gold mentions only when surface form and coarse type are identical."""
    scoped = coarsen_gold(gold)
    clusters_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, mention in enumerate(scoped["mentions"]):
        clusters_by_key[(mention["tags"][0], mention["text"].strip())].append(index)
    return _prediction_from_clusters(scoped, list(clusters_by_key.values()))


def _acronym(value: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", value)
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens).casefold()


def exact_alias_prediction(gold: dict[str, Any]) -> dict[str, Any]:
    """Cluster Gold mentions using only normalized surface and acronym aliases."""
    scoped = coarsen_gold(gold)
    mentions = scoped["mentions"]
    union_find = _UnionFind(len(mentions))
    keys: dict[tuple[str, str], int] = {}
    acronyms: dict[tuple[str, str], int] = {}
    for index, mention in enumerate(mentions):
        entity_type = mention["tags"][0]
        text = mention["text"]
        signature = identity_signature(text, entity_type)
        key = (entity_type, signature)
        if key in keys:
            union_find.union(index, keys[key])
        else:
            keys[key] = index
        acronym = _acronym(text)
        if acronym:
            acronym_key = (entity_type, acronym)
            if acronym_key in acronyms:
                union_find.union(index, acronyms[acronym_key])
            else:
                acronyms[acronym_key] = index

    for index, mention in enumerate(mentions):
        entity_type = mention["tags"][0]
        normalized = normalize_name(mention["text"])
        acronym_index = acronyms.get((entity_type, normalized))
        if acronym_index is not None:
            union_find.union(index, acronym_index)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(mentions)):
        clusters[union_find.find(index)].append(index)
    ordered_clusters = sorted(clusters.values(), key=lambda cluster: min(cluster))
    return _prediction_from_clusters(scoped, ordered_clusters)


def has_plausible_candidate(
    snapshot: TKGSnapshot,
    candidate: SemanticEntityCandidate,
    *,
    lexical_threshold: float = 0.55,
) -> bool:
    """Avoid remote resolution when no same-type name is lexically plausible."""
    candidate_forms = [candidate.name, *candidate.aliases]
    for entity in snapshot.entities:
        if entity.status != "active" or entity.entity_type != candidate.entity_type:
            continue
        for left in candidate_forms:
            left_signature = identity_signature(left, candidate.entity_type)
            for right in [entity.canonical_name, *entity.aliases]:
                right_signature = identity_signature(right, entity.entity_type)
                if not left_signature or not right_signature:
                    continue
                if left_signature == right_signature:
                    return True
                if min(len(left_signature), len(right_signature)) >= 3 and (
                    left_signature in right_signature or right_signature in left_signature
                ):
                    return True
                if (
                    SequenceMatcher(None, left_signature, right_signature).ratio()
                    >= lexical_threshold
                ):
                    return True
    return False


def gold_alias_candidates(
    document: dict[str, Any],
    *,
    context_window: int = 220,
) -> tuple[list[SemanticEntityCandidate], list[int]]:
    """Run Exact+Alias first and return one resolver candidate per remaining cluster."""
    alias_prediction = exact_alias_prediction(document)
    mentions = alias_prediction["mentions"]
    content = str(document.get("content") or "")
    clusters: dict[int, list[dict[str, Any]]] = {}
    for mention in mentions:
        clusters.setdefault(int(mention["concept"]), []).append(mention)
    candidates = []
    for concept_id in range(len(alias_prediction["concepts"])):
        members = clusters[concept_id]
        representative = max(members, key=lambda item: len(item["text"]))
        surfaces = list(dict.fromkeys(item["text"] for item in members))
        contexts = list(
            dict.fromkeys(
                content[
                    max(0, int(item["begin"]) - context_window) : min(
                        len(content), int(item["end"]) + context_window
                    )
                ]
                for item in members
            )
        )
        candidates.append(
            SemanticEntityCandidate(
                name=representative["text"],
                entity_type=representative["tags"][0],
                aliases=[value for value in surfaces if value != representative["text"]],
                description="\n...\n".join(contexts[:3])[:1_200],
            )
        )
    return candidates, [int(mention["concept"]) for mention in mentions]


def _literal_spans(content: str, value: str) -> list[tuple[int, int]]:
    cleaned = value.strip().strip("\"“”'")
    if not cleaned:
        return []
    pattern = re.compile(rf"(?<!\w){re.escape(cleaned)}(?!\w)", re.IGNORECASE)
    return [(match.start(), match.end()) for match in pattern.finditer(content)]


def semantic_prediction(
    document: dict[str, Any],
    semantic: SemanticExtractionBatch,
    *,
    resolved_entity_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Align graph entity names back to DWIE character spans."""
    resolved_entity_ids = resolved_entity_ids or {}
    content = str(document.get("content") or "")
    candidates = sorted(semantic.entities, key=lambda item: len(item.name), reverse=True)
    span_owner: dict[tuple[int, int], tuple[str, str, str]] = {}
    for candidate in candidates:
        entity_type = normalize_predicted_type(candidate.entity_type)
        if entity_type is None:
            continue
        candidate_key = "|".join(
            (normalize_name(candidate.name), normalize_name(candidate.entity_type))
        )
        entity_id = resolved_entity_ids.get(candidate_key, candidate_key)
        for surface in [candidate.name, *candidate.aliases]:
            for span in _literal_spans(content, surface):
                span_owner.setdefault(span, (entity_id, entity_type, content[slice(*span)]))

    cluster_keys = sorted(
        {value[0] for value in span_owner.values()},
        key=lambda key: min(span for span, value in span_owner.items() if value[0] == key),
    )
    concept_ids = {key: index for index, key in enumerate(cluster_keys)}
    concepts = []
    for key in cluster_keys:
        values = [value for value in span_owner.values() if value[0] == key]
        representative = max(values, key=lambda item: len(item[2]))
        concepts.append(
            {
                "concept": concept_ids[key],
                "text": representative[2],
                "tags": [representative[1]],
            }
        )
    mentions = [
        {
            "begin": begin,
            "end": end,
            "text": value[2],
            "concept": concept_ids[value[0]],
            "tags": [value[1]],
        }
        for (begin, end), value in sorted(span_owner.items())
    ]
    return {
        "id": document["id"],
        "content": content,
        "tags": document.get("tags", []),
        "mentions": mentions,
        "concepts": concepts,
        "relations": [],
    }


def load_official_evaluator(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("dwie_official_evaluation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import DWIE evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_entity_predictions(
    predictions: list[dict[str, Any]],
    gold_documents: list[dict[str, Any]],
    *,
    official_evaluator_path: Path,
) -> dict[str, Any]:
    module = load_official_evaluator(official_evaluator_path)
    evaluator = module.EvaluatorDWIE()
    predictions_by_id = {item["id"]: item for item in predictions}
    for raw_gold in gold_documents:
        gold = coarsen_gold(raw_gold)
        prediction = predictions_by_id.get(raw_gold["id"])
        if prediction is None:
            prediction = {
                "id": raw_gold["id"],
                "content": raw_gold.get("content", ""),
                "tags": raw_gold.get("tags", []),
                "mentions": [],
                "concepts": [],
                "relations": [],
            }
        evaluator.add(prediction, gold)

    coref_values = [
        evaluator.coref_muc.get_f1(),
        evaluator.coref_bcubed.get_f1(),
        evaluator.coref_ceafe.get_f1(),
    ]
    return {
        "scope": "DWIE shared coarse types: Person, Organization, Location",
        "documents": len(gold_documents),
        "gold_mentions": sum(len(coarsen_gold(item)["mentions"]) for item in gold_documents),
        "predicted_mentions": sum(len(item.get("mentions", [])) for item in predictions),
        "mention": {
            "precision": evaluator.tags_mention.get_pr(),
            "recall": evaluator.tags_mention.get_re(),
            "f1": evaluator.tags_mention.get_f1(),
        },
        "entity_soft": {
            "precision": evaluator.tags_soft.get_pr(),
            "recall": evaluator.tags_soft.get_re(),
            "f1": evaluator.tags_soft.get_f1(),
        },
        "entity_hard": {
            "precision": evaluator.tags_hard.get_pr(),
            "recall": evaluator.tags_hard.get_re(),
            "f1": evaluator.tags_hard.get_f1(),
        },
        "coreference": {
            "muc_f1": coref_values[0],
            "bcubed_f1": coref_values[1],
            "ceafe_f1": coref_values[2],
            "conll_f1": sum(coref_values) / len(coref_values),
        },
    }


def read_documents(path: Path) -> list[dict[str, Any]]:
    return [json.loads(file.read_text(encoding="utf-8")) for file in sorted(path.glob("*.json"))]
