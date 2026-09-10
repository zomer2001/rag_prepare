from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tiktoken

from rag_graph_demo.evaluation.native_context import NativeContextRegistry

ContextVariant = Literal["raw", "graph", "evolution", "native"]
CONTEXT_VARIANTS: tuple[ContextVariant, ...] = ("raw", "graph", "evolution", "native")
QA_CONTEXT_PROFILE_VERSION = "gold_qa_context_v2_native_fixed_token_budget"
EVOLUTION_EDGE_TYPES = {
    "CREATES",
    "UPDATES",
    "TERMINATES",
    "SUPERSEDES",
    "REPLACES",
    "DENIES",
    "CONFIRMS",
}


class TokenCounter:
    """Deterministic budget counter shared by every answer context."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._encoding = (
            None if encoding_name == "unicode_chars_v1" else tiktoken.get_encoding(encoding_name)
        )

    def count(self, value: str) -> int:
        if self._encoding is None:
            return len(value)
        return len(self._encoding.encode(value))

    def truncate(self, value: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self._encoding is None:
            return value[:max_tokens].rstrip()
        tokens = self._encoding.encode(value)
        if len(tokens) <= max_tokens:
            return value
        return self._encoding.decode(tokens[:max_tokens]).rstrip()


@dataclass(frozen=True)
class HydratedEvidence:
    evidence_id: str
    document_id: str | None
    rank: int
    text: str


@dataclass(frozen=True)
class BuiltAnswerContext:
    text: str
    token_count: int
    raw_token_count: int
    structured_token_count: int
    evidence_ids: tuple[str, ...]
    graph_stats: dict[str, int]
    structured_context_origin: str


class EvidenceHydrator:
    """Restore full evidence text from immutable corpus and index artifacts."""

    def __init__(
        self,
        *,
        external_corpus_path: Path,
        light_chunk_store_path: Path | None = None,
        event_store_path: Path | None = None,
    ) -> None:
        corpus = json.loads(external_corpus_path.read_text(encoding="utf-8"))
        if not isinstance(corpus, list):
            raise ValueError("External corpus must be a JSON array")
        self.external_chunks = {
            str(item["chunk_id"]): str(item.get("text") or "")
            for item in corpus
            if item.get("chunk_id")
        }
        self.light_chunks: dict[str, str] = {}
        if light_chunk_store_path and light_chunk_store_path.is_file():
            stored = json.loads(light_chunk_store_path.read_text(encoding="utf-8"))
            self.light_chunks = {
                str(key): str(value.get("content") or "")
                for key, value in stored.items()
                if isinstance(value, dict)
            }
        self.event_mentions: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.segment_mentions: dict[str, list[str]] = defaultdict(list)
        if event_store_path and event_store_path.is_file():
            event_store = json.loads(event_store_path.read_text(encoding="utf-8"))
            for mention in (event_store.get("mentions") or {}).values():
                if not isinstance(mention, dict):
                    continue
                evidence = str(mention.get("evidence") or "").strip()
                if not evidence:
                    continue
                event_id = str(mention.get("event_id") or "")
                document_id = str(mention.get("document_id") or "")
                segment_id = str(mention.get("source_segment_id") or "")
                self.event_mentions[(event_id, document_id)].append(evidence)
                if segment_id:
                    self.segment_mentions[segment_id].append(evidence)

    def hydrate_row(self, row: dict[str, Any], *, top_k: int) -> list[HydratedEvidence]:
        result: list[HydratedEvidence] = []
        for rank, item in enumerate((row.get("retrieved") or [])[:top_k], start=1):
            metadata = item.get("metadata") or {}
            passages = metadata.get("passages") or []
            texts: list[str] = []
            for passage in passages:
                if isinstance(passage, dict):
                    texts.extend(self._passage_texts(passage, item))
            if not texts:
                texts.extend(self._chunk_texts(str(item.get("chunk_id") or "")))
            if not texts:
                fallback = str(item.get("text") or "").strip()
                if fallback:
                    texts.append(fallback)
            texts = _deduplicate_contained_texts(texts)
            document_id = (
                str(item["document_id"])
                if item.get("document_id") not in (None, "")
                else None
            )
            evidence_id = f"E{rank}"
            result.append(
                HydratedEvidence(
                    evidence_id=evidence_id,
                    document_id=document_id,
                    rank=rank,
                    text="\n\n".join(texts).strip(),
                )
            )
        return result

    def _passage_texts(
        self,
        passage: dict[str, Any],
        parent: dict[str, Any],
    ) -> list[str]:
        chunk_id = str(passage.get("chunk_id") or "")
        direct = self._chunk_texts(chunk_id)
        if direct:
            return direct
        metadata = passage.get("metadata") or {}
        if str(passage.get("channel") or "") == "event":
            event_id = str(metadata.get("event_id") or "")
            document_id = str(parent.get("document_id") or "")
            values = self.event_mentions.get((event_id, document_id), [])
            if values:
                return values
            return self.segment_mentions.get(chunk_id, [])
        return []

    def _chunk_texts(self, chunk_id: str) -> list[str]:
        if not chunk_id:
            return []
        if chunk_id in self.light_chunks:
            return [self.light_chunks[chunk_id]]
        if chunk_id in self.external_chunks:
            return [self.external_chunks[chunk_id]]
        external_id = re.sub(r"-chunk-\d+$", "", chunk_id)
        if external_id in self.external_chunks:
            return [self.external_chunks[external_id]]
        return self.segment_mentions.get(chunk_id, [])


class AnswerContextBuilder:
    def __init__(
        self,
        hydrator: EvidenceHydrator,
        *,
        token_counter: TokenCounter | None = None,
        max_context_tokens: int = 16_000,
        max_evidence_tokens: int = 1_200,
        structured_token_share: float = 0.25,
        native_contexts: NativeContextRegistry | None = None,
    ) -> None:
        if max_context_tokens <= 0 or max_evidence_tokens <= 0:
            raise ValueError("Context and per-evidence token budgets must be positive")
        if not 0 <= structured_token_share < 1:
            raise ValueError("structured_token_share must be in [0, 1)")
        self.hydrator = hydrator
        self.tokens = token_counter or TokenCounter()
        self.max_context_tokens = max_context_tokens
        self.max_evidence_tokens = max_evidence_tokens
        self.structured_token_share = structured_token_share
        self.native_contexts = native_contexts or NativeContextRegistry()

    def build(
        self,
        row: dict[str, Any],
        *,
        method: str = "ours",
        variant: ContextVariant,
        top_k: int,
    ) -> BuiltAnswerContext:
        if variant not in CONTEXT_VARIANTS:
            raise ValueError(f"Unsupported context variant: {variant}")
        evidence = self.hydrator.hydrate_row(row, top_k=top_k)
        structured_header = "## Structured evidence\n"
        raw_header = "## Original evidence\n"
        section_separator = "\n\n"
        graph_text = ""
        graph_stats: dict[str, int] = {}
        structured_context_origin = "none"
        if variant != "raw":
            structured_section_budget = int(
                self.max_context_tokens * self.structured_token_share
            )
            if variant == "native" and method in self.native_contexts.views:
                native_sections, graph_stats = self.native_contexts.render_sections(
                    method,
                    row,
                    top_k=top_k,
                )
                available = max(
                    0,
                    structured_section_budget - self.tokens.count(structured_header),
                )
                per_view_budget = available // max(1, len(native_sections))
                graph_text = "\n\n".join(
                    self.tokens.truncate(section, per_view_budget)
                    for section in native_sections
                )
                structured_context_origin = "reconstructed_from_index_by_retrieved_sources"
            else:
                include_evolution = variant == "evolution" or (
                    variant == "native" and method in {"evolink", "d1_norerank"}
                )
                graph_text, graph_stats = build_structured_context(
                    row,
                    evidence,
                    include_evolution=include_evolution,
                )
                structured_context_origin = "retrieval_result_evidence_graph"
            graph_text = self.tokens.truncate(
                graph_text,
                max(0, structured_section_budget - self.tokens.count(structured_header)),
            )
            if not graph_text:
                structured_context_origin = "no_source_aligned_structure_available"
        structured_tokens = self.tokens.count(graph_text) if graph_text else 0
        fixed_text = raw_header
        if graph_text:
            fixed_text = structured_header + graph_text + section_separator + raw_header
        raw_budget = max(0, self.max_context_tokens - self.tokens.count(fixed_text))
        raw_text, _ = self._raw_context(evidence, raw_budget)
        sections = []
        if graph_text:
            sections.append(structured_header + graph_text)
        if raw_text:
            sections.append(raw_header + raw_text)
        context = section_separator.join(sections)
        context = self.tokens.truncate(context, self.max_context_tokens)
        present_ids = set(re.findall(r"\bE\d+\b", context))
        return BuiltAnswerContext(
            text=context,
            token_count=self.tokens.count(context),
            raw_token_count=self.tokens.count(raw_text),
            structured_token_count=structured_tokens,
            evidence_ids=tuple(
                item.evidence_id
                for item in evidence
                if item.evidence_id in present_ids
            ),
            graph_stats=graph_stats,
            structured_context_origin=structured_context_origin,
        )

    def _raw_context(
        self,
        evidence: list[HydratedEvidence],
        budget: int,
    ) -> tuple[str, list[str]]:
        blocks: list[str] = []
        used_ids: list[str] = []
        remaining = budget
        for item in evidence:
            if remaining <= 0:
                break
            separator = "\n\n" if blocks else ""
            separator_tokens = self.tokens.count(separator)
            header = f"[{item.evidence_id}; document_id={item.document_id or 'unknown'}]\n"
            header_tokens = self.tokens.count(header)
            content_budget = min(
                self.max_evidence_tokens,
                remaining - separator_tokens - header_tokens,
            )
            if content_budget <= 0:
                break
            content = self.tokens.truncate(item.text, content_budget)
            if not content:
                continue
            block = header + content
            block_tokens = self.tokens.count(block)
            if block_tokens + separator_tokens > remaining:
                block = self.tokens.truncate(block, remaining - separator_tokens)
                block_tokens = self.tokens.count(block)
            blocks.append(block)
            used_ids.append(item.evidence_id)
            remaining -= block_tokens + separator_tokens
        return "\n\n".join(blocks), used_ids


def build_structured_context(
    row: dict[str, Any],
    evidence: list[HydratedEvidence],
    *,
    include_evolution: bool,
) -> tuple[str, dict[str, int]]:
    graphs = []
    seen_graphs: set[str] = set()
    for item in (row.get("retrieved") or [])[: len(evidence)]:
        graph = ((item.get("metadata") or {}).get("evidence_graph") or {})
        signature = json.dumps(graph, ensure_ascii=False, sort_keys=True)
        if signature not in seen_graphs and (graph.get("nodes") or graph.get("edges")):
            graphs.append(graph)
            seen_graphs.add(signature)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[Any, ...], dict[str, Any]] = {}
    for graph in graphs:
        for node in graph.get("nodes") or []:
            if isinstance(node, dict) and node.get("id"):
                nodes[str(node["id"])] = node
        for edge in graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            key = (
                edge.get("source"),
                edge.get("target"),
                edge.get("type"),
                edge.get("document_id"),
            )
            edges[key] = edge

    doc_to_evidence = {
        item.document_id: item.evidence_id for item in evidence if item.document_id
    }
    allowed_docs = set(doc_to_evidence)
    selected: set[str] = set()
    for node_id, node in nodes.items():
        sources = set(str(value) for value in node.get("source_document_ids") or [])
        if sources & allowed_docs:
            selected.add(node_id)
    for edge in edges.values():
        if str(edge.get("document_id") or "") in allowed_docs:
            selected.update((str(edge.get("source") or ""), str(edge.get("target") or "")))
    association_types = {"SUBJECT", "OBJECT", "ACTOR"}
    core = set(selected)
    for edge in edges.values():
        if str(edge.get("type") or "") not in association_types:
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in core and nodes.get(target, {}).get("type") == "entity":
            selected.add(target)
        if target in core and nodes.get(source, {}).get("type") == "entity":
            selected.add(source)
    selected.discard("")
    filtered_edges = [
        edge
        for edge in edges.values()
        if str(edge.get("source") or "") in selected
        and str(edge.get("target") or "") in selected
    ]

    def source_refs(node: dict[str, Any], edge: dict[str, Any] | None = None) -> str:
        sources = [str(value) for value in node.get("source_document_ids") or []]
        if edge and edge.get("document_id"):
            sources.append(str(edge["document_id"]))
        refs = [
            doc_to_evidence[value]
            for value in dict.fromkeys(sources)
            if value in doc_to_evidence
        ]
        return ",".join(refs) or "unknown"

    incoming: dict[tuple[str, str], list[str]] = defaultdict(list)
    outgoing: dict[tuple[str, str], list[str]] = defaultdict(list)
    for edge in filtered_edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        edge_type = str(edge.get("type") or "")
        incoming[(target, edge_type)].append(source)
        outgoing[(source, edge_type)].append(target)

    entities = [nodes[node_id] for node_id in selected if nodes[node_id].get("type") == "entity"]
    assertions = [
        nodes[node_id] for node_id in selected if nodes[node_id].get("type") == "assertion"
    ]
    events = [nodes[node_id] for node_id in selected if nodes[node_id].get("type") == "event"]
    entities.sort(key=lambda node: (str(node.get("label") or ""), str(node.get("id") or "")))
    assertions.sort(key=lambda node: (source_refs(node), str(node.get("id") or "")))
    events.sort(key=lambda node: (source_refs(node), str(node.get("id") or "")))

    sections: list[str] = []
    if include_evolution:
        lines = ["### Evolution effects", "event | effect | assertion | effective_time | source"]
        for edge in filtered_edges:
            edge_type = str(edge.get("type") or "")
            if edge_type not in EVOLUTION_EDGE_TYPES:
                continue
            source = nodes[str(edge.get("source"))]
            target = nodes[str(edge.get("target"))]
            lines.append(
                " | ".join(
                    (
                        _field(source.get("label")),
                        edge_type,
                        _field(target.get("label") or target.get("predicate")),
                        _field(edge.get("effective_time")),
                        source_refs(target, edge),
                    )
                )
            )
        if len(lines) > 2:
            sections.append("\n".join(lines))

    if entities:
        lines = ["### Entities", "entity | type | source"]
        for node in entities:
            lines.append(
                " | ".join(
                    (
                        _field(node.get("label")),
                        _field(node.get("entity_type")),
                        source_refs(node),
                    )
                )
            )
        sections.append("\n".join(lines))
    if assertions:
        lines = [
            "### Relations and assertions",
            "subject | relation | object | status | valid | source",
        ]
        for node in assertions:
            node_id = str(node["id"])
            subjects = [
                _field(nodes[value].get("label"))
                for value in incoming[(node_id, "SUBJECT")]
            ]
            objects = [
                _field(nodes[value].get("label"))
                for value in outgoing[(node_id, "OBJECT")]
            ]
            valid = "..".join(
                (
                    _field(node.get("valid_from"), default=""),
                    _field(node.get("valid_to"), default=""),
                )
            ).strip(".")
            lines.append(
                " | ".join(
                    (
                        ", ".join(subjects) or "unknown",
                        _field(node.get("predicate") or node.get("label")),
                        ", ".join(objects) or "unknown",
                        _field(node.get("status")),
                        valid or "unknown",
                        source_refs(node),
                    )
                )
            )
        sections.append("\n".join(lines))
    if events:
        lines = ["### Events", "event | type | time | actors | objects | status | source"]
        for node in events:
            node_id = str(node["id"])
            actors = [_field(nodes[value].get("label")) for value in outgoing[(node_id, "ACTOR")]]
            objects = [_field(nodes[value].get("label")) for value in outgoing[(node_id, "OBJECT")]]
            event_time = node.get("start_time") or node.get("end_time")
            lines.append(
                " | ".join(
                    (
                        _field(node.get("label")),
                        _field(node.get("event_type")),
                        _field(event_time),
                        ", ".join(actors) or "unknown",
                        ", ".join(objects) or "unknown",
                        _field(node.get("status")),
                        source_refs(node),
                    )
                )
            )
        sections.append("\n".join(lines))
    effect_count = sum(
        str(edge.get("type") or "") in EVOLUTION_EDGE_TYPES for edge in filtered_edges
    )
    return "\n\n".join(sections), {
        "entities": len(entities),
        "assertions": len(assertions),
        "events": len(events),
        "edges": len(filtered_edges),
        "evolution_edges": effect_count,
    }


def normalize_short_answer(value: str) -> str:
    lowered = value.casefold()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    tokens = [token for token in without_punctuation.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def answer_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_short_answer(prediction) == normalize_short_answer(reference))


def answer_token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_short_answer(prediction).split()
    gold = normalize_short_answer(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    common = Counter(predicted) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def aggregate_answer_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "query_count": len(values),
            "exact_match": (
                sum(float(row.get("exact_match") or 0) for row in values) / len(values)
                if values
                else 0.0
            ),
            "token_f1": (
                sum(float(row.get("token_f1") or 0) for row in values) / len(values)
                if values
                else 0.0
            ),
        }
        for field in (
            "elapsed_ms",
            "api_elapsed_ms",
            "queue_wait_ms",
            "context_tokens",
            "prompt_tokens",
            "completion_tokens",
            "citation_id_validity",
        ):
            observed = [float(row[field]) for row in values if row.get(field) is not None]
            if observed:
                result[f"mean_{field}"] = sum(observed) / len(observed)
        return result

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        by_type[str(row.get("question_type") or "unknown")].append(row)
    result = summarize(successful)
    result.update(
        {
            "error_count": len(rows) - len(successful),
            "by_question_type": {
                key: summarize(values) for key, values in sorted(by_type.items())
            },
        }
    )
    return result


def _field(value: Any, *, default: str = "unknown") -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.replace("|", "/") if text else default


def _deduplicate_contained_texts(values: list[str]) -> list[str]:
    cleaned = [re.sub(r"\n{3,}", "\n\n", value).strip() for value in values if value.strip()]
    result: list[str] = []
    for value in cleaned:
        if any(value in existing for existing in result):
            continue
        result = [existing for existing in result if existing not in value]
        result.append(value)
    return result
