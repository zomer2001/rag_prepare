#!/usr/bin/env python3
"""Build fair answer contexts, generate answers, and score short-answer QA."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from json_repair import repair_json
from openai import AsyncOpenAI

from rag_graph_demo.config import get_settings
from rag_graph_demo.evaluation.answering import (
    CONTEXT_VARIANTS,
    QA_CONTEXT_PROFILE_VERSION,
    AnswerContextBuilder,
    EvidenceHydrator,
    TokenCounter,
    aggregate_answer_scores,
    answer_exact_match,
    answer_token_f1,
)
from rag_graph_demo.evaluation.native_context import (
    NativeContextRegistry,
    SourceAlignedNativeGraph,
)

ANSWER_PROMPT_PROFILE_VERSION = "gold_short_answer_v1_grounded_citations"
ANSWER_SYSTEM_PROMPT = """You are a grounded question-answering system. Answer only from the
provided context and do not use outside knowledge. Return one JSON object with exactly these
keys: short_answer (string), explanation (string), citations (array of evidence IDs). Keep
short_answer minimal: use Yes or No for binary questions, otherwise return only the requested
name, date, number, or short phrase. Briefly justify the answer in explanation. Cite only IDs
such as E1 that occur in the supplied context. If the context is insufficient, use an empty
short_answer and explain what is missing. Return JSON only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "generate", "score", "all"), default="all")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--external-corpus", type=Path, required=True)
    parser.add_argument("--light-chunks", type=Path)
    parser.add_argument("--event-store", type=Path)
    parser.add_argument(
        "--native-graph",
        action="append",
        default=[],
        metavar="METHOD:LABEL=GRAPHML",
        help="Offline GraphML view used to reconstruct a method's native structured context.",
    )
    parser.add_argument(
        "--native-snapshot",
        action="append",
        default=[],
        metavar="METHOD:LABEL=JSON",
        help="Offline JSON graph snapshot used to reconstruct native structured context.",
    )
    parser.add_argument(
        "--native-event-store",
        action="append",
        default=[],
        metavar="METHOD:LABEL=JSON",
        help="Internal event store used to reconstruct event-native context without querying.",
    )
    parser.add_argument(
        "--native-full-docs",
        action="append",
        default=[],
        metavar="METHOD:LABEL=JSON",
        help="Optional full-document store for matching retrieved text to graph source IDs.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="METHOD:VARIANT=JSONL",
        help=(
            "Retrieval result and context variant. Repeat for each run, for example "
            "--input hybrid:raw=path.jsonl."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-context-tokens", type=int, default=16_000)
    parser.add_argument("--max-evidence-tokens", type=int, default=1_200)
    parser.add_argument("--structured-token-share", type=float, default=0.25)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_summary_tables(output_dir: Path, summary: dict[str, dict[str, Any]]) -> None:
    labels = {
        "naive__raw": ("Naive RAG", "Raw evidence"),
        "hybrid__raw": ("Hybrid", "Raw evidence"),
        "light_entity__raw": ("LightRAG Entity", "Raw evidence"),
        "light_entity__native": ("LightRAG Entity", "Native entity/relation tables"),
        "ours__raw": ("Ours", "Raw evidence"),
        "ours__native": ("Ours", "Native entity/event views"),
        "evolink__raw": ("EvoLink", "Raw evidence"),
        "evolink__native": ("EvoLink", "Native evidence graph"),
        "hipporag__raw": ("HippoRAG 2", "Raw evidence"),
        "eventrag__raw": ("EventRAG", "Raw evidence"),
        "eventrag__native": ("EventRAG", "Native entity/event tables"),
        "e2rag__raw": ("E²RAG", "Raw evidence"),
        "e2rag__native": ("E²RAG", "Native entity/event views"),
        "graphrag__raw": ("GraphRAG Local", "Raw evidence"),
        "d1_norerank__raw": ("EvoNeed-D1", "Raw evidence"),
        "d1_norerank__native": ("EvoNeed-D1", "Native evidence graph"),
        "d1_norerank__graph": ("EvoNeed-D1", "Entity/relation/event tables"),
        "d1_norerank__evolution": ("EvoNeed-D1", "Evolution structure"),
    }
    type_keys = {
        "comparison_em": "comparison_query",
        "inference_em": "inference_query",
        "temporal_em": "temporal_query",
    }
    raw_values = summary.get("d1_norerank__raw")
    raw_em = float(raw_values.get("exact_match") or 0) if raw_values else None

    def rounded(value: Any) -> float:
        return round(float(value or 0), 6)

    rows: list[dict[str, Any]] = []
    for run_id, values in summary.items():
        method, context = labels.get(run_id, (run_id, "unknown"))
        by_type = values.get("by_question_type") or {}
        row: dict[str, Any] = {
            "run_id": run_id,
            "method": method,
            "answer_context": context,
            "queries": int(values.get("query_count") or 0),
            "exact_match": rounded(values.get("exact_match")),
            "token_f1": rounded(values.get("token_f1")),
            "delta_em_vs_d1_raw": (
                rounded(float(values.get("exact_match") or 0) - raw_em)
                if raw_em is not None
                else None
            ),
        }
        for column, question_type in type_keys.items():
            row[column] = rounded(
                (by_type.get(question_type) or {}).get("exact_match")
            )
        row.update(
            {
                "citation_id_validity": rounded(
                    values.get("mean_citation_id_validity")
                ),
                "mean_context_tokens": rounded(values.get("mean_context_tokens")),
                "mean_prompt_tokens": rounded(values.get("mean_prompt_tokens")),
                "mean_completion_tokens": rounded(values.get("mean_completion_tokens")),
                "errors": int(values.get("error_count") or 0),
            }
        )
        rows.append(row)

    columns = [
        "run_id",
        "method",
        "answer_context",
        "queries",
        "exact_match",
        "token_f1",
        "delta_em_vs_d1_raw",
        "comparison_em",
        "inference_em",
        "temporal_em",
        "citation_id_validity",
        "mean_context_tokens",
        "mean_prompt_tokens",
        "mean_completion_tokens",
        "errors",
    ]
    csv_path = output_dir / "answer_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def percent(value: Any) -> str:
        if value is None:
            return "n.a."
        return f"{float(value):.1%}"

    markdown = [
        "# Gold500 answer evaluation pilot\n",
        "Gold500 first 100 questions; Top-20 evidence; DeepSeek `deepseek-v4-flash`; "
        "temperature 0; thinking disabled; maximum context 16,000 tokens.\n",
        "| Method | Answer context | N | EM | Token F1 | ΔEM vs D1 raw | "
        "Comparison EM | Inference EM | Temporal EM | Citation ID validity | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (-item["exact_match"], item["run_id"])):
        markdown.append(
            "| {method} | {answer_context} | {queries} | {em} | {f1} | {delta} | "
            "{comparison} | {inference} | {temporal} | {citation} | {errors} |".format(
                **row,
                em=percent(row["exact_match"]),
                f1=percent(row["token_f1"]),
                delta=(
                    f"{row['delta_em_vs_d1_raw']:+.1%}"
                    if row["delta_em_vs_d1_raw"] is not None
                    else "n.a."
                ),
                comparison=percent(row["comparison_em"]),
                inference=percent(row["inference_em"]),
                temporal=percent(row["temporal_em"]),
                citation=percent(row["citation_id_validity"]),
            )
        )
    best_em = max((row["exact_match"] for row in rows), default=0)
    best_labels = [
        f"{row['method']} ({row['answer_context']})"
        for row in rows
        if row["exact_match"] == best_em
    ]
    graph_delta: float | None = next(
        (
            row["delta_em_vs_d1_raw"]
            for row in rows
            if row["run_id"] == "d1_norerank__graph"
        ),
        None,
    )
    evolution_delta: float | None = next(
        (
            row["delta_em_vs_d1_raw"]
            for row in rows
            if row["run_id"] == "d1_norerank__evolution"
        ),
        None,
    )
    markdown.extend(
        [
            "\n## Current reading\n",
            f"- Highest EM: {', '.join(best_labels)} at {percent(best_em)}.\n",
        ]
    )
    if graph_delta is not None and evolution_delta is not None:
        markdown.append(
            f"- EvoNeed-D1 generic graph tables change EM by {graph_delta:+.1%} "
            "versus its raw-evidence context; the evolution structure changes it by "
            f"{evolution_delta:+.1%}.\n"
        )
    markdown.extend(
        [
            "- Treat this as a pilot result. Validate on all 500 questions before "
            "making a paper claim.\n",
            "- Latency is omitted for legacy batches that recorded time spent waiting "
            "in the concurrency queue. New batches record clean API latency separately.\n",
        ]
    )
    (output_dir / "answer_summary.md").write_text("\n".join(markdown), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_or_validate_manifest(path: Path, expected: dict[str, Any]) -> None:
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise RuntimeError(
                f"Existing manifest does not match this answer run: {path}. "
                "Use a new output directory."
            )
        return
    path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_input_specs(values: list[str]) -> list[tuple[str, str, Path]]:
    result: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for value in values:
        left, separator, raw_path = value.partition("=")
        method, variant_separator, variant = left.partition(":")
        if not separator or not variant_separator or not method or not variant or not raw_path:
            raise ValueError(f"Invalid --input value: {value}")
        if variant not in CONTEXT_VARIANTS:
            raise ValueError(f"Invalid context variant in --input: {value}")
        run_id = f"{method}__{variant}"
        if run_id in seen:
            raise ValueError(f"Duplicate answer run: {run_id}")
        seen.add(run_id)
        result.append((method, variant, Path(raw_path).resolve()))
    return result


def parse_native_resource(value: str) -> tuple[str, str, Path]:
    left, separator, raw_path = value.partition("=")
    method, label_separator, label = left.partition(":")
    if not separator or not label_separator or not method or not label or not raw_path:
        raise ValueError(f"Invalid native resource specification: {value}")
    return method, label, Path(raw_path).resolve()


def build_native_contexts(
    args: argparse.Namespace,
) -> tuple[NativeContextRegistry, list[dict[str, Any]]]:
    full_docs: dict[tuple[str, str], Path] = {}
    for value in args.native_full_docs:
        method, label, path = parse_native_resource(value)
        full_docs[(method, label)] = path
    registry = NativeContextRegistry()
    resources: list[dict[str, Any]] = []
    definitions = (
        [("graphml", value) for value in args.native_graph]
        + [("snapshot", value) for value in args.native_snapshot]
        + [("event_store", value) for value in args.native_event_store]
    )
    for resource_type, value in definitions:
        method, label, path = parse_native_resource(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing native graph resource: {path}")
        full_docs_path = full_docs.get((method, label))
        if full_docs_path is not None and not full_docs_path.is_file():
            raise FileNotFoundError(f"Missing native full-doc store: {full_docs_path}")
        if resource_type == "graphml":
            view = SourceAlignedNativeGraph.from_graphml(
                path,
                label=label,
                full_docs_path=full_docs_path,
            )
        elif resource_type == "snapshot":
            view = SourceAlignedNativeGraph.from_snapshot(
                path,
                label=label,
                full_docs_path=full_docs_path,
            )
        else:
            view = SourceAlignedNativeGraph.from_event_store(path, label=label)
        registry.add(method, view)
        resource = {
            "method": method,
            "label": label,
            "type": resource_type,
            "path": str(path),
            "sha256": sha256(path),
        }
        if full_docs_path is not None:
            resource["full_docs_path"] = str(full_docs_path)
            resource["full_docs_sha256"] = sha256(full_docs_path)
        resources.append(resource)
    requested_native = {
        method
        for method, variant, _ in parse_input_specs(args.input)
        if variant == "native"
    }
    external_native = requested_native & {"light_entity", "eventrag", "e2rag"}
    missing = external_native - set(registry.views)
    if missing:
        raise ValueError(
            "Native graph resources are required for: " + ", ".join(sorted(missing))
        )
    return registry, resources


def load_dataset_queries(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "queries.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset queries: {path}")
    result = []
    for index, row in enumerate(load_jsonl(path)):
        question = row.get("question") or row.get("query")
        if not question or row.get("answer") is None:
            raise ValueError(f"Query {index} lacks a question or reference answer")
        result.append(
            {
                "query_id": str(row.get("query_id") or row.get("id") or f"q-{index:05d}"),
                "question": str(question),
                "question_type": str(row.get("question_type") or row.get("type") or "unknown"),
                "reference_answer": str(row["answer"]),
            }
        )
    return result


def write_context_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_context_tokens: int,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_id"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for run_id, values in grouped.items():
        count = len(values)
        with_structure = sum(int(row["structured_tokens"]) > 0 for row in values)
        origins: dict[str, int] = {}
        for row in values:
            origin = str(row["structured_context_origin"])
            origins[origin] = origins.get(origin, 0) + 1
        summary.append(
            {
                "run_id": run_id,
                "method": values[0]["method"],
                "variant": values[0]["variant"],
                "queries": count,
                "structured_queries": with_structure,
                "structured_coverage": with_structure / count if count else 0.0,
                "mean_context_tokens": sum(row["context_tokens"] for row in values) / count,
                "mean_structured_tokens": sum(row["structured_tokens"] for row in values) / count,
                "mean_raw_tokens": sum(row["raw_tokens"] for row in values) / count,
                "max_context_tokens": max(row["context_tokens"] for row in values),
                "origins": origins,
            }
        )
    summary.sort(key=lambda row: str(row["run_id"]))
    (output_dir / "context_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    columns = [
        "run_id",
        "method",
        "variant",
        "queries",
        "structured_queries",
        "structured_coverage",
        "mean_context_tokens",
        "mean_structured_tokens",
        "mean_raw_tokens",
        "max_context_tokens",
        "origins",
    ]
    with (output_dir / "context_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in summary:
            output = dict(row)
            output["origins"] = json.dumps(row["origins"], ensure_ascii=False, sort_keys=True)
            writer.writerow(output)
    markdown = [
        "# Gold500 native answer-context summary\n",
        "No retrieval or answer-model calls were made. Structured information is source-aligned "
        "to each method's already retrieved evidence, with a fixed "
        f"{max_context_tokens:,}-token total budget.\n",
        "| Run | Questions | Structured coverage | Mean total tokens | "
        "Mean structured tokens | Mean raw tokens | Maximum tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        markdown.append(
            "| {run_id} | {queries} | {coverage:.1%} | {total:.1f} | "
            "{structured:.1f} | {raw:.1f} | {maximum} |".format(
                run_id=row["run_id"],
                queries=row["queries"],
                coverage=row["structured_coverage"],
                total=row["mean_context_tokens"],
                structured=row["mean_structured_tokens"],
                raw=row["mean_raw_tokens"],
                maximum=row["max_context_tokens"],
            )
        )
    markdown.extend(
        [
            "\n## Interpretation\n",
            "- `reconstructed_from_index_by_retrieved_sources`: offline reconstruction from the "
            "method's existing index, restricted to sources it retrieved.\n",
            "- `retrieval_result_evidence_graph`: structure already attached to the saved "
            "retrieval result.\n",
            "- Missing structure means that no source-aligned graph structure was available for "
            "that query; its answer context contains raw evidence only.\n",
        ]
    )
    (output_dir / "context_summary.md").write_text("\n".join(markdown), encoding="utf-8")


def prepare_contexts(args: argparse.Namespace, specs: list[tuple[str, str, Path]]) -> Path:
    if not specs:
        raise ValueError("At least one --input is required for context preparation")
    queries = load_dataset_queries(args.dataset_dir.resolve())
    if args.limit is not None:
        queries = queries[: args.limit]
    query_by_id = {row["query_id"]: row for row in queries}
    hydrator = EvidenceHydrator(
        external_corpus_path=args.external_corpus.resolve(),
        light_chunk_store_path=args.light_chunks.resolve() if args.light_chunks else None,
        event_store_path=args.event_store.resolve() if args.event_store else None,
    )
    native_contexts, native_resources = build_native_contexts(args)
    builder = AnswerContextBuilder(
        hydrator,
        token_counter=TokenCounter(args.tokenizer),
        max_context_tokens=args.max_context_tokens,
        max_evidence_tokens=args.max_evidence_tokens,
        structured_token_share=args.structured_token_share,
        native_contexts=native_contexts,
    )
    contexts_path = args.output_dir.resolve() / "contexts.jsonl"
    temporary = contexts_path.with_suffix(".jsonl.tmp")
    contexts_path.parent.mkdir(parents=True, exist_ok=True)
    expected_rows = 0
    context_summary_rows: list[dict[str, Any]] = []
    with temporary.open("w", encoding="utf-8") as handle:
        for method, variant, path in specs:
            rows = {str(row["query_id"]): row for row in load_jsonl(path)}
            missing = set(query_by_id) - set(rows)
            if missing:
                raise ValueError(f"{method}:{variant} is missing {len(missing)} requested queries")
            for query in queries:
                built = builder.build(
                    rows[query["query_id"]],
                    method=method,
                    variant=variant,
                    top_k=args.top_k,
                )
                output = {
                    "run_id": f"{method}__{variant}",
                    "method": method,
                    "variant": variant,
                    **query,
                    "context": built.text,
                    "context_tokens": built.token_count,
                    "raw_tokens": built.raw_token_count,
                    "structured_tokens": built.structured_token_count,
                    "evidence_ids": list(built.evidence_ids),
                    "graph_stats": built.graph_stats,
                    "structured_context_origin": built.structured_context_origin,
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                context_summary_rows.append(output)
                expected_rows += 1
    temporary.replace(contexts_path)
    write_context_summary(
        contexts_path.parent,
        context_summary_rows,
        max_context_tokens=args.max_context_tokens,
    )
    manifest = {
        "context_profile": QA_CONTEXT_PROFILE_VERSION,
        "answer_prompt_profile": ANSWER_PROMPT_PROFILE_VERSION,
        "dataset_queries_sha256": sha256(args.dataset_dir.resolve() / "queries.jsonl"),
        "external_corpus_sha256": sha256(args.external_corpus.resolve()),
        "inputs": [
            {
                "method": method,
                "variant": variant,
                "path": str(path),
                "sha256": sha256(path),
            }
            for method, variant, path in specs
        ],
        "native_resources": native_resources,
        "query_count_per_run": len(queries),
        "context_count": expected_rows,
        "top_k": args.top_k,
        "max_context_tokens": args.max_context_tokens,
        "max_evidence_tokens": args.max_evidence_tokens,
        "structured_token_share": args.structured_token_share,
        "tokenizer": args.tokenizer,
    }
    (contexts_path.parent / "context_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {expected_rows} answer contexts: {contexts_path}")
    return contexts_path


def extract_answer(content: str) -> dict[str, Any]:
    value = repair_json(content, return_objects=True)
    if not isinstance(value, dict):
        raise ValueError("Answer model did not return a JSON object")
    citations = value.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    return {
        "short_answer": str(value.get("short_answer") or "").strip(),
        "explanation": str(value.get("explanation") or "").strip(),
        "citations": [str(item).strip() for item in citations if str(item).strip()],
    }


async def generate_answers(args: argparse.Namespace, contexts_path: Path) -> Path:
    settings = get_settings()
    api_key = os.getenv("DEEPSEEK_API_KEY") or settings.deepseek_api_key
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for --stage generate")
    model = args.model or settings.deepseek_chat_model
    base_url = args.base_url or settings.deepseek_base_url
    answers_path = args.output_dir.resolve() / "answers.jsonl"
    run_manifest = {
        "answer_prompt_profile": ANSWER_PROMPT_PROFILE_VERSION,
        "contexts_sha256": sha256(contexts_path),
        "model": model,
        "base_url": base_url,
        "thinking": "disabled",
        "temperature": 0,
        "max_output_tokens": args.max_output_tokens,
    }
    manifest_path = answers_path.parent / "answer_run_manifest.json"
    if answers_path.is_file() and not manifest_path.is_file():
        raise RuntimeError(
            f"Existing answers have no run manifest: {answers_path}. Use a new output directory."
        )
    write_or_validate_manifest(manifest_path, run_manifest)
    completed: set[tuple[str, str]] = set()
    if answers_path.is_file():
        completed = {
            (str(row["run_id"]), str(row["query_id"]))
            for row in load_jsonl(answers_path)
            if not row.get("error")
        }
    rows = [
        row
        for row in load_jsonl(contexts_path)
        if (str(row["run_id"]), str(row["query_id"])) not in completed
    ]
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    output_lock = asyncio.Lock()

    async def run_one(row: dict[str, Any]) -> None:
        queued_at = perf_counter()
        execution_started: float | None = None
        api_elapsed_ms = 0.0
        content = ""
        finish_reason = ""
        try:
            total_prompt_tokens = 0
            total_completion_tokens = 0
            parse_attempts = max(1, settings.llm_max_retries)
            for parse_attempt in range(1, parse_attempts + 1):
                async with semaphore:
                    if execution_started is None:
                        execution_started = perf_counter()
                    request_started = perf_counter()
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": (
                                    f"Question: {row['question']}\n\n"
                                    f"Context:\n{row['context']}\n\nReturn JSON only."
                                ),
                            },
                        ],
                        response_format={"type": "json_object"},
                        extra_body={"thinking": {"type": "disabled"}},
                        temperature=0,
                        max_tokens=args.max_output_tokens,
                    )
                    api_elapsed_ms += (perf_counter() - request_started) * 1000
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason or ""
                usage = response.usage
                total_prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                total_completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                try:
                    parsed = extract_answer(content)
                    break
                except (TypeError, ValueError):
                    if parse_attempt >= parse_attempts:
                        raise
                    await asyncio.sleep(min(2 ** (parse_attempt - 1), 4))
            output = {
                "run_id": row["run_id"],
                "method": row["method"],
                "variant": row["variant"],
                "query_id": row["query_id"],
                "question": row["question"],
                "question_type": row["question_type"],
                "reference_answer": row["reference_answer"],
                **parsed,
                "available_evidence_ids": row["evidence_ids"],
                "context_tokens": row["context_tokens"],
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "elapsed_ms": (perf_counter() - (execution_started or queued_at)) * 1000,
                "api_elapsed_ms": api_elapsed_ms,
                "queue_wait_ms": ((execution_started or queued_at) - queued_at) * 1000,
                "model": model,
                "parse_attempts": parse_attempt,
                "finish_reason": finish_reason,
            }
        except Exception as exc:
            if args.fail_fast:
                raise
            output = {
                "run_id": row["run_id"],
                "method": row["method"],
                "variant": row["variant"],
                "query_id": row["query_id"],
                "question": row["question"],
                "question_type": row["question_type"],
                "reference_answer": row["reference_answer"],
                "elapsed_ms": (perf_counter() - (execution_started or queued_at)) * 1000,
                "api_elapsed_ms": api_elapsed_ms,
                "queue_wait_ms": ((execution_started or queued_at) - queued_at) * 1000,
                "model": model,
                "error": f"{type(exc).__name__}: {exc}",
                "finish_reason": finish_reason,
                "response_content_excerpt": content[:1000],
            }
        async with output_lock:
            append_jsonl(answers_path, output)

    await asyncio.gather(*(run_one(row) for row in rows))
    print(f"Generated {len(rows)} new answers: {answers_path}")
    return answers_path


def score_answers(args: argparse.Namespace, answers_path: Path) -> Path:
    # A resumed run may contain an earlier failed attempt followed by a success.
    # Keep only the latest record for each method/query pair when scoring.
    latest_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(answers_path):
        latest_rows[(str(row["run_id"]), str(row["query_id"]))] = row
    rows = list(latest_rows.values())
    scored = []
    for row in rows:
        output = dict(row)
        if not row.get("error"):
            prediction = str(row.get("short_answer") or "")
            reference = str(row.get("reference_answer") or "")
            available = set(str(item) for item in row.get("available_evidence_ids") or [])
            citations = [str(item) for item in row.get("citations") or []]
            output.update(
                {
                    "exact_match": answer_exact_match(prediction, reference),
                    "token_f1": answer_token_f1(prediction, reference),
                    "citation_id_validity": (
                        sum(item in available for item in citations) / len(citations)
                        if citations
                        else 0.0
                    ),
                }
            )
        scored.append(output)
    scored_path = args.output_dir.resolve() / "answers_scored.jsonl"
    scored_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored),
        encoding="utf-8",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        grouped.setdefault(str(row["run_id"]), []).append(row)
    summary = {
        run_id: aggregate_answer_scores(values)
        for run_id, values in sorted(grouped.items())
    }
    summary_path = args.output_dir.resolve() / "answer_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_tables(args.output_dir.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Scored answers written to: {summary_path}")
    return summary_path


async def run(args: argparse.Namespace) -> int:
    if args.top_k <= 0 or args.concurrency <= 0 or args.max_output_tokens <= 0:
        raise ValueError("top-k, concurrency, and max-output-tokens must be positive")
    specs = parse_input_specs(args.input)
    output_dir = args.output_dir.resolve()
    contexts_path = output_dir / "contexts.jsonl"
    answers_path = output_dir / "answers.jsonl"
    if args.stage in {"prepare", "all"}:
        contexts_path = prepare_contexts(args, specs)
    elif not contexts_path.is_file():
        raise FileNotFoundError(f"Missing prepared contexts: {contexts_path}")
    if args.stage in {"generate", "all"}:
        answers_path = await generate_answers(args, contexts_path)
    elif args.stage == "score" and not answers_path.is_file():
        raise FileNotFoundError(f"Missing generated answers: {answers_path}")
    if args.stage in {"score", "all"}:
        score_answers(args, answers_path)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
