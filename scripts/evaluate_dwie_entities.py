#!/usr/bin/env python3
"""Run the small DWIE entity extraction and resolution benchmark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rag_graph_demo.config import Settings
from rag_graph_demo.domain.tkg import (
    EntityRecord,
    ResolutionAction,
    SemanticEntityCandidate,
    SemanticExtractionBatch,
    TKGSnapshot,
)
from rag_graph_demo.evaluation.dwie_entities import (
    coarsen_gold,
    evaluate_entity_predictions,
    exact_alias_prediction,
    exact_match_prediction,
    gold_alias_candidates,
    has_plausible_candidate,
    read_documents,
    semantic_prediction,
    singleton_prediction,
)
from rag_graph_demo.ingestion.source_mapper import SourceMapper
from rag_graph_demo.lightrag_engine.engine import LightRAGEngine
from rag_graph_demo.lightrag_engine.local_models import build_embedding_func
from rag_graph_demo.services.entity_resolution_service import (
    EntityResolutionService,
    OntologyTypeMapper,
    normalize_name,
)
from rag_graph_demo.storage.document_registry import DocumentRegistry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "data" / "benchmarks" / "extraction" / "dwie_test_seed20260817_n20"
OFFICIAL_EVALUATOR = (
    ROOT
    / "data"
    / "benchmarks"
    / "_downloads"
    / "extraction"
    / "dwie"
    / "src"
    / "dwie_evaluation.py"
)
METHODS = {
    "official",
    "all_singleton",
    "exact_match",
    "exact_alias",
    "eventrag_gold_mentions",
    "lightrag",
    "light_entity",
    "ours_no_llm",
    "ours_entity",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return read_documents(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Prediction file must contain a JSON list: {path}")
    return value


def _candidate_key(candidate: SemanticEntityCandidate) -> str:
    return "|".join((normalize_name(candidate.name), normalize_name(candidate.entity_type)))


async def _resolve_candidates(
    document_id: str,
    candidates: list[SemanticEntityCandidate],
    resolver: EntityResolutionService,
    *,
    gate_unrelated: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Apply the current resolver sequentially within one DWIE document."""
    snapshot = TKGSnapshot(ontology_version="dwie-shared-v1")
    resolved_ids: dict[str, str] = {}
    mention_entity_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        proposal = None
        if not gate_unrelated or has_plausible_candidate(snapshot, candidate):
            proposal = (await resolver.resolve_batch(snapshot, [candidate]))[0]
        selected = None
        if proposal is not None and proposal.action == ResolutionAction.MATCH_EXISTING:
            selected = next(
                (
                    entity
                    for entity in snapshot.entities
                    if entity.entity_id == proposal.selected_entity_id and entity.status == "active"
                ),
                None,
            )
        if selected is None:
            selected = EntityRecord(
                entity_id=f"{document_id}:entity:{index}",
                canonical_name=candidate.name,
                entity_type=(
                    proposal.normalized_type if proposal is not None else candidate.entity_type
                ),
                aliases=list(candidate.aliases),
                description=candidate.description,
                ontology_version="dwie-shared-v1",
            )
            snapshot.entities.append(selected)
        elif (
            normalize_name(candidate.name) != normalize_name(selected.canonical_name)
            and candidate.name not in selected.aliases
        ):
            selected.aliases.append(candidate.name)
        resolved_ids[_candidate_key(candidate)] = selected.entity_id
        mention_entity_ids.append(selected.entity_id)
    return resolved_ids, mention_entity_ids


def _prediction_from_gold_resolution(
    document: dict[str, Any], mention_entity_ids: list[str]
) -> dict[str, Any]:
    scoped = coarsen_gold(document)
    mentions = scoped["mentions"]
    if len(mentions) != len(mention_entity_ids):
        raise ValueError("Resolved mention count does not match the scoped DWIE mentions")
    ordered_ids = list(dict.fromkeys(mention_entity_ids))
    concept_ids = {entity_id: index for index, entity_id in enumerate(ordered_ids)}
    concepts = []
    for entity_id in ordered_ids:
        member_indexes = [
            index for index, value in enumerate(mention_entity_ids) if value == entity_id
        ]
        representative = max(member_indexes, key=lambda index: len(mentions[index]["text"]))
        concepts.append(
            {
                "concept": concept_ids[entity_id],
                "text": mentions[representative]["text"],
                "tags": mentions[representative]["tags"],
            }
        )
    return {
        **scoped,
        "concepts": concepts,
        "mentions": [
            {**mention, "concept": concept_ids[entity_id]}
            for mention, entity_id in zip(mentions, mention_entity_ids, strict=True)
        ],
    }


def _build_engine(
    base_settings: Settings,
    runtime_dir: Path,
    method: str,
    *,
    controlled: bool,
) -> LightRAGEngine:
    method_dir = runtime_dir / method
    settings = base_settings.model_copy(
        update={
            "lightrag_working_dir": method_dir / "lightrag",
            "lightrag_workspace": f"dwie_{method}_{runtime_dir.name}",
            "lightrag_summary_language": "English",
            "embedding_provider": "zhipu",
            "event_extraction_enabled": False,
        }
    )
    registry = DocumentRegistry(method_dir / "documents.json")
    return LightRAGEngine(
        settings,
        SourceMapper(registry),
        embedding_func=build_embedding_func(settings),
        extraction_profile="light_entity" if controlled else "default",
    )


async def _semantic_batches(
    documents: list[dict[str, Any]],
    engine: LightRAGEngine,
    cache_dir: Path,
) -> dict[str, SemanticExtractionBatch]:
    batches: dict[str, SemanticExtractionBatch] = {}
    pending = []
    for document in documents:
        cache_path = cache_dir / f"{document['id']}.json"
        if cache_path.exists():
            batches[document["id"]] = SemanticExtractionBatch.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
        else:
            pending.append(
                (
                    document["id"],
                    document["content"],
                    f"{document['id']}__{document['id']}.txt",
                )
            )
    if pending:
        await engine.index_documents(pending)
    for document in documents:
        if document["id"] in batches:
            continue
        batch = await engine.get_document_semantic_extraction(document["id"])
        batches[document["id"]] = batch
        _write_json(cache_dir / f"{document['id']}.json", batch.model_dump(mode="json"))
    return batches


async def _run_lightrag(
    documents: list[dict[str, Any]],
    settings: Settings,
    runtime_dir: Path,
) -> list[dict[str, Any]]:
    method = "lightrag"
    engine = _build_engine(settings, runtime_dir, method, controlled=False)
    batches = await _semantic_batches(documents, engine, runtime_dir / method / "semantic")
    return [semantic_prediction(document, batches[document["id"]]) for document in documents]


async def _run_light_entity(
    documents: list[dict[str, Any]],
    settings: Settings,
    runtime_dir: Path,
) -> list[dict[str, Any]]:
    # Share the controlled extraction index with ours_entity so this ablation
    # adds no second indexing cost.
    method = "ours_entity"
    engine = _build_engine(settings, runtime_dir, method, controlled=True)
    batches = await _semantic_batches(documents, engine, runtime_dir / method / "semantic")
    return [semantic_prediction(document, batches[document["id"]]) for document in documents]


def _build_resolver(
    settings: Settings,
    engine: LightRAGEngine,
    *,
    allow_llm: bool,
) -> EntityResolutionService:
    return EntityResolutionService(
        OntologyTypeMapper(settings.entity_type_mapping_file),
        engine.embed_texts,
        engine.complete_extraction,
        candidate_limit=settings.entity_resolution_candidate_limit,
        embedding_threshold=settings.entity_resolution_embedding_threshold,
        auto_match_threshold=settings.entity_resolution_auto_match_threshold,
        llm_threshold=(settings.entity_resolution_llm_threshold if allow_llm else 2.0),
    )


async def _run_gold_resolution(
    documents: list[dict[str, Any]],
    settings: Settings,
    runtime_dir: Path,
    *,
    method: str,
    allow_llm: bool,
) -> list[dict[str, Any]]:
    engine = _build_engine(settings, runtime_dir, method, controlled=True)
    resolver = _build_resolver(settings, engine, allow_llm=allow_llm)
    predictions = []
    cache_dir = runtime_dir / method / "resolution" / "gold_mentions"
    for position, document in enumerate(documents, start=1):
        cache_path = cache_dir / f"{document['id']}.json"
        if cache_path.exists():
            prediction = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            candidates, mention_cluster_ids = gold_alias_candidates(document)
            _, cluster_entity_ids = await _resolve_candidates(
                document["id"], candidates, resolver, gate_unrelated=True
            )
            mention_entity_ids = [
                cluster_entity_ids[cluster_id] for cluster_id in mention_cluster_ids
            ]
            prediction = _prediction_from_gold_resolution(document, mention_entity_ids)
            _write_json(cache_path, prediction)
        predictions.append(prediction)
        print(f"{method} resolution {position}/{len(documents)}: {document['id']}")
    return predictions


async def _run_eventrag_gold_resolution(
    documents: list[dict[str, Any]],
    settings: Settings,
    runtime_dir: Path,
    *,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate EventRAG's Top-1 description-similarity merge on Gold mentions.

    Exact+Alias is a shared pre-clustering step.  Each remaining cluster is
    represented by the same contextual description used by our resolver.  The
    adapter then follows EventRAG's resolution rule: compare the new entity to
    previously indexed descriptions and merge it into the Top-1 entity only
    when cosine similarity is above the configured threshold.
    """
    method = "eventrag_gold_mentions"
    embedding_func = build_embedding_func(settings)
    predictions = []
    cache_dir = runtime_dir / method / f"threshold_{similarity_threshold:.3f}"
    for position, document in enumerate(documents, start=1):
        cache_path = cache_dir / f"{document['id']}.json"
        if cache_path.exists():
            prediction = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            candidates, mention_cluster_ids = gold_alias_candidates(document)
            if candidates:
                descriptions = [candidate.description or candidate.name for candidate in candidates]
                vectors = np.asarray(
                    await embedding_func(descriptions, context="document"),
                    dtype=np.float32,
                )
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-12)
            else:
                vectors = np.empty((0, 0), dtype=np.float32)

            # EventRAG compares against already indexed entity descriptions.
            # A merge points at the existing canonical entity; it does not
            # update that entity's stored vector during this decision loop.
            canonical_indexes: list[int] = []
            cluster_entity_ids: list[str] = []
            for index, _candidate in enumerate(candidates):
                selected_index = index
                if canonical_indexes:
                    similarities = vectors[canonical_indexes] @ vectors[index]
                    best_position = int(np.argmax(similarities))
                    if float(similarities[best_position]) > similarity_threshold:
                        selected_index = canonical_indexes[best_position]
                if selected_index == index:
                    canonical_indexes.append(index)
                cluster_entity_ids.append(f"{document['id']}:entity:{selected_index}")

            mention_entity_ids = [
                cluster_entity_ids[cluster_id] for cluster_id in mention_cluster_ids
            ]
            prediction = _prediction_from_gold_resolution(document, mention_entity_ids)
            _write_json(cache_path, prediction)
        predictions.append(prediction)
        print(f"{method} resolution {position}/{len(documents)}: {document['id']}")
    return predictions


async def _run_ours_entity(
    documents: list[dict[str, Any]],
    settings: Settings,
    runtime_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method = "ours_entity"
    engine = _build_engine(settings, runtime_dir, method, controlled=True)
    batches = await _semantic_batches(documents, engine, runtime_dir / method / "semantic")
    resolver = _build_resolver(settings, engine, allow_llm=True)
    end_to_end = []
    gold_mentions = []
    resolution_cache = runtime_dir / method / "resolution"
    for position, document in enumerate(documents, start=1):
        semantic = batches[document["id"]]
        end_to_end_path = resolution_cache / "end_to_end" / f"{document['id']}.json"
        if end_to_end_path.exists():
            end_to_end_prediction = json.loads(end_to_end_path.read_text(encoding="utf-8"))
        else:
            resolved_ids, _ = await _resolve_candidates(
                document["id"], list(semantic.entities), resolver
            )
            end_to_end_prediction = semantic_prediction(
                document, semantic, resolved_entity_ids=resolved_ids
            )
            _write_json(end_to_end_path, end_to_end_prediction)
        end_to_end.append(end_to_end_prediction)

        gold_path = resolution_cache / "gold_mentions" / f"{document['id']}.json"
        if gold_path.exists():
            gold_prediction = json.loads(gold_path.read_text(encoding="utf-8"))
        else:
            candidates, mention_cluster_ids = gold_alias_candidates(document)
            _, cluster_entity_ids = await _resolve_candidates(
                document["id"], candidates, resolver, gate_unrelated=True
            )
            mention_entity_ids = [
                cluster_entity_ids[cluster_id] for cluster_id in mention_cluster_ids
            ]
            gold_prediction = _prediction_from_gold_resolution(document, mention_entity_ids)
            _write_json(gold_path, gold_prediction)
        gold_mentions.append(gold_prediction)
        print(f"ours_entity resolution {position}/{len(documents)}: {document['id']}")
    return end_to_end, gold_mentions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--methods",
        default="exact_alias,lightrag,ours_entity",
        help=f"Comma-separated: {','.join(sorted(METHODS))}",
    )
    parser.add_argument("--official-predictions", type=Path)
    parser.add_argument(
        "--eventrag-entity-threshold",
        type=float,
        default=0.9,
        help="Top-1 description cosine threshold used by EventRAG (default: 0.9)",
    )
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(sorted(unknown))}")
    sample_dir = args.sample_dir.resolve()
    documents = read_documents(sample_dir / "gold")
    if not documents:
        raise FileNotFoundError(
            f"No sample documents found in {sample_dir / 'gold'}. "
            "Run scripts/prepare_dwie_entity_benchmark.py first."
        )
    settings = Settings()
    manifest_digest = hashlib.sha256((sample_dir / "manifest.json").read_bytes()).hexdigest()[:12]
    evaluation_profile = {
        "version": "dwie_entity_v3_cumulative_resolution",
        "language": "English",
        "deepseek_extraction_model": settings.deepseek_extraction_model,
        "embedding_provider": "zhipu",
        "zhipu_embedding_model": settings.zhipu_embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "light_entity_max_entities": settings.light_entity_max_entities,
        "light_entity_max_relationships": settings.light_entity_max_relationships,
        "entity_resolution_embedding_threshold": (settings.entity_resolution_embedding_threshold),
        "entity_resolution_auto_match_threshold": (settings.entity_resolution_auto_match_threshold),
        "entity_resolution_llm_threshold": settings.entity_resolution_llm_threshold,
        "entity_resolution_precluster": "exact_alias",
        "entity_resolution_lexical_gate": 0.55,
    }
    run_digest = hashlib.sha256(
        (
            manifest_digest
            + "\n"
            + json.dumps(evaluation_profile, sort_keys=True, ensure_ascii=False)
        ).encode()
    ).hexdigest()[:12]
    runtime_dir = (args.runtime_dir or sample_dir / "runtime" / run_digest).resolve()
    output_dir = (args.output_dir or sample_dir / "results" / run_digest).resolve()
    if {"lightrag", "light_entity", "ours_entity"} & set(methods) and not (
        settings.deepseek_api_key
    ):
        raise ValueError("DEEPSEEK_API_KEY is required")
    if {
        "lightrag",
        "light_entity",
        "eventrag_gold_mentions",
        "ours_no_llm",
        "ours_entity",
    } & set(methods) and not settings.zhipu_api_key:
        raise ValueError("ZHIPU_API_KEY is required")

    summary_path = output_dir / "summary.json"
    previous_methods: dict[str, Any] = {}
    if summary_path.exists():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous_summary.get("run_digest") == run_digest:
            previous_methods = dict(previous_summary.get("methods") or {})

    summary: dict[str, Any] = {
        "dataset": "DWIE",
        "sample_dir": str(sample_dir),
        "manifest_digest": manifest_digest,
        "run_digest": run_digest,
        "evaluation_profile": evaluation_profile,
        "notes": {
            "scope": "Person/Organization/Location shared coarse types only",
            "gold_mention_track": (
                "Exact+Alias, EventRAG adapter, and ours_entity/gold_mentions receive "
                "identical Gold mentions and coarse types; only clustering is evaluated."
            ),
            "end_to_end_track": (
                "LightRAG and ours_entity/end_to_end extract names, align literal surface "
                "occurrences, then evaluate mention typing and clustering."
            ),
        },
        # A later invocation may add a cheap baseline or official predictions
        # without discarding API-backed results already produced for this run.
        "methods": previous_methods,
    }

    gold_baselines = {
        "all_singleton": singleton_prediction,
        "exact_match": exact_match_prediction,
        "exact_alias": exact_alias_prediction,
    }
    for method, predict in gold_baselines.items():
        if method not in methods:
            continue
        predictions = [predict(document) for document in documents]
        prediction_dir = output_dir / "predictions" / method
        for prediction in predictions:
            _write_json(prediction_dir / f"{prediction['id']}.json", prediction)
        summary["methods"][method] = {
            "track": "gold_mentions",
            "metrics": evaluate_entity_predictions(
                predictions,
                documents,
                official_evaluator_path=OFFICIAL_EVALUATOR,
            ),
        }

    if "lightrag" in methods:
        predictions = await _run_lightrag(documents, settings, runtime_dir)
        prediction_dir = output_dir / "predictions" / "lightrag"
        for prediction in predictions:
            _write_json(prediction_dir / f"{prediction['id']}.json", prediction)
        summary["methods"]["lightrag"] = {
            "track": "end_to_end",
            "metrics": evaluate_entity_predictions(
                predictions,
                documents,
                official_evaluator_path=OFFICIAL_EVALUATOR,
            ),
        }

    if "light_entity" in methods:
        predictions = await _run_light_entity(documents, settings, runtime_dir)
        prediction_dir = output_dir / "predictions" / "light_entity"
        for prediction in predictions:
            _write_json(prediction_dir / f"{prediction['id']}.json", prediction)
        summary["methods"]["light_entity"] = {
            "track": "end_to_end",
            "metrics": evaluate_entity_predictions(
                predictions,
                documents,
                official_evaluator_path=OFFICIAL_EVALUATOR,
            ),
        }

    if "ours_no_llm" in methods:
        predictions = await _run_gold_resolution(
            documents,
            settings,
            runtime_dir,
            method="ours_no_llm",
            allow_llm=False,
        )
        prediction_dir = output_dir / "predictions" / "ours_no_llm"
        for prediction in predictions:
            _write_json(prediction_dir / f"{prediction['id']}.json", prediction)
        summary["methods"]["ours_no_llm"] = {
            "track": "gold_mentions",
            "metrics": evaluate_entity_predictions(
                predictions,
                documents,
                official_evaluator_path=OFFICIAL_EVALUATOR,
            ),
        }

    if "eventrag_gold_mentions" in methods:
        predictions = await _run_eventrag_gold_resolution(
            documents,
            settings,
            runtime_dir,
            similarity_threshold=args.eventrag_entity_threshold,
        )
        prediction_dir = output_dir / "predictions" / "eventrag_gold_mentions"
        for prediction in predictions:
            _write_json(prediction_dir / f"{prediction['id']}.json", prediction)
        summary["methods"]["eventrag_gold_mentions"] = {
            "track": "gold_mentions",
            "adapter": "EventRAG Top-1 description-embedding merge after shared Exact+Alias",
            "similarity_threshold": args.eventrag_entity_threshold,
            "metrics": evaluate_entity_predictions(
                predictions,
                documents,
                official_evaluator_path=OFFICIAL_EVALUATOR,
            ),
        }

    if "ours_entity" in methods:
        end_to_end, gold_mentions = await _run_ours_entity(documents, settings, runtime_dir)
        for track, predictions in (
            ("end_to_end", end_to_end),
            ("gold_mentions", gold_mentions),
        ):
            prediction_dir = output_dir / "predictions" / "ours_entity" / track
            for prediction in predictions:
                _write_json(prediction_dir / f"{prediction['id']}.json", prediction)
            summary["methods"][f"ours_entity/{track}"] = {
                "track": track,
                "metrics": evaluate_entity_predictions(
                    predictions,
                    documents,
                    official_evaluator_path=OFFICIAL_EVALUATOR,
                ),
            }

    if "official" in methods:
        if args.official_predictions is None:
            summary["methods"]["official"] = {
                "status": "predictions_required",
                "reason": (
                    "The DWIE repository publishes training code but no pretrained Joint BERT "
                    "checkpoint. Pass --official-predictions after training or obtaining "
                    "predictions."
                ),
            }
        else:
            predictions = [
                coarsen_gold(item)
                for item in _read_predictions(args.official_predictions.resolve())
            ]
            summary["methods"]["official"] = {
                "track": "end_to_end",
                "metrics": evaluate_entity_predictions(
                    predictions,
                    documents,
                    official_evaluator_path=OFFICIAL_EVALUATOR,
                ),
            }

    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to {output_dir}")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
