from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


def _source_ids(value: Any) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in re.split(r"<SEP>|\|\|", str(value or ""))
        if part.strip()
    )


def _clean(value: Any, *, limit: int = 360, default: str = "unknown") -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip('"')
    text = text.replace("|", "/")
    if not text:
        return default
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _normalise_content(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


@dataclass(frozen=True)
class NativeGraphNode:
    node_id: str
    name: str
    node_type: str
    description: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class NativeGraphEdge:
    source: str
    target: str
    relation: str
    description: str
    source_ids: tuple[str, ...]


class SourceAlignedNativeGraph:
    """Read-only graph view aligned to evidence that a method already retrieved."""

    def __init__(
        self,
        *,
        label: str,
        nodes: list[NativeGraphNode],
        edges: list[NativeGraphEdge],
        full_docs_path: Path | None = None,
        max_nodes: int = 60,
        max_edges: int = 60,
    ) -> None:
        self.label = label
        self.nodes = {node.node_id: node for node in nodes}
        self.edges = edges
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.node_ids_by_source: dict[str, set[str]] = defaultdict(set)
        self.edge_indexes_by_source: dict[str, set[int]] = defaultdict(set)
        for node in nodes:
            for source_id in node.source_ids:
                self.node_ids_by_source[source_id].add(node.node_id)
        for index, edge in enumerate(edges):
            for source_id in edge.source_ids:
                self.edge_indexes_by_source[source_id].add(index)
        self.known_sources = set(self.node_ids_by_source) | set(self.edge_indexes_by_source)
        self.sources_by_content: dict[str, set[str]] = defaultdict(set)
        self.sources_by_prefix: dict[str, set[str]] = defaultdict(set)
        if full_docs_path and full_docs_path.is_file():
            stored = json.loads(full_docs_path.read_text(encoding="utf-8"))
            for item_id, value in stored.items():
                content = value.get("content") if isinstance(value, dict) else value
                normalised = _normalise_content(content)
                if not normalised:
                    continue
                candidates = {str(item_id)}
                if str(item_id).startswith("doc-"):
                    candidates.add("chunk-" + str(item_id)[4:])
                matched = candidates & self.known_sources
                self.sources_by_content[normalised].update(matched)
                self.sources_by_prefix[normalised[:500]].update(matched)

    @classmethod
    def from_graphml(
        cls,
        path: Path,
        *,
        label: str,
        full_docs_path: Path | None = None,
        max_nodes: int = 60,
        max_edges: int = 60,
    ) -> SourceAlignedNativeGraph:
        graph = nx.read_graphml(path)
        nodes = [
            NativeGraphNode(
                node_id=str(node_id),
                name=_display_node_name(str(node_id)),
                node_type=str(data.get("entity_type") or data.get("node_type") or "entity"),
                description=str(data.get("description") or ""),
                source_ids=_source_ids(data.get("source_id")),
            )
            for node_id, data in graph.nodes(data=True)
        ]
        edges = [
            NativeGraphEdge(
                source=str(source),
                target=str(target),
                relation=str(data.get("keywords") or data.get("relation") or "related_to"),
                description=str(data.get("description") or ""),
                source_ids=_source_ids(data.get("source_id")),
            )
            for source, target, data in graph.edges(data=True)
        ]
        return cls(
            label=label,
            nodes=nodes,
            edges=edges,
            full_docs_path=full_docs_path,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    @classmethod
    def from_snapshot(
        cls,
        path: Path,
        *,
        label: str,
        full_docs_path: Path | None = None,
        max_nodes: int = 60,
        max_edges: int = 60,
    ) -> SourceAlignedNativeGraph:
        value = json.loads(path.read_text(encoding="utf-8"))
        nodes = [
            NativeGraphNode(
                node_id=str(item["id"]),
                name=str(item.get("name") or item["id"]),
                node_type=str(item.get("type") or "entity"),
                description=str(item.get("description") or ""),
                source_ids=_source_ids(item.get("source_ids") or item.get("source_id")),
            )
            for item in value.get("nodes") or []
        ]
        edges = [
            NativeGraphEdge(
                source=str(item["source"]),
                target=str(item["target"]),
                relation=str(item.get("relation") or "related_to"),
                description=str(item.get("description") or ""),
                source_ids=_source_ids(item.get("source_ids") or item.get("source_id")),
            )
            for item in value.get("edges") or []
        ]
        return cls(
            label=label,
            nodes=nodes,
            edges=edges,
            full_docs_path=full_docs_path,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    @classmethod
    def from_event_store(
        cls,
        path: Path,
        *,
        label: str,
        max_nodes: int = 60,
        max_edges: int = 60,
    ) -> SourceAlignedNativeGraph:
        value = json.loads(path.read_text(encoding="utf-8"))
        events = value.get("events") or {}
        nodes: dict[str, NativeGraphNode] = {}
        edges: list[NativeGraphEdge] = []
        for event_id, event in events.items():
            sources = tuple(str(item) for item in event.get("source_document_ids") or [])
            nodes[str(event_id)] = NativeGraphNode(
                node_id=str(event_id),
                name=str(event.get("summary") or event.get("action") or event_id),
                node_type=str(event.get("event_type") or "event"),
                description=str(event.get("result") or event.get("summary") or ""),
                source_ids=sources,
            )
            for role, values in (
                ("actor", event.get("actors") or []),
                ("object", event.get("objects") or []),
            ):
                for entity_name in values:
                    entity_id = f"entity::{entity_name}"
                    current = nodes.get(entity_id)
                    combined_sources = sources
                    if current is not None:
                        combined_sources = tuple(dict.fromkeys((*current.source_ids, *sources)))
                    nodes[entity_id] = NativeGraphNode(
                        node_id=entity_id,
                        name=str(entity_name),
                        node_type="entity",
                        description="",
                        source_ids=combined_sources,
                    )
                    edges.append(
                        NativeGraphEdge(
                            source=str(event_id),
                            target=entity_id,
                            relation=role,
                            description="",
                            source_ids=sources,
                        )
                    )
        for relation in (value.get("relations") or {}).values():
            source = str(relation.get("source_event_id") or "")
            target = str(relation.get("target_event_id") or "")
            if source not in nodes or target not in nodes:
                continue
            sources = tuple(
                dict.fromkeys((*nodes[source].source_ids, *nodes[target].source_ids))
            )
            edges.append(
                NativeGraphEdge(
                    source=source,
                    target=target,
                    relation=str(relation.get("relation_type") or "related_to"),
                    description=", ".join(relation.get("shared_entities") or []),
                    source_ids=sources,
                )
            )
        return cls(
            label=label,
            nodes=list(nodes.values()),
            edges=edges,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def render_sections(
        self,
        row: dict[str, Any],
        *,
        top_k: int,
    ) -> tuple[list[str], dict[str, int]]:
        source_ranks: dict[str, int] = {}
        for rank, item in enumerate((row.get("retrieved") or [])[:top_k], start=1):
            candidates = {str(item.get("chunk_id") or "")}
            metadata = item.get("metadata") or {}
            candidates.add(str(item.get("document_id") or ""))
            for passage in metadata.get("passages") or []:
                if isinstance(passage, dict):
                    candidates.add(str(passage.get("chunk_id") or ""))
            normalised_text = _normalise_content(item.get("text"))
            candidates.update(self.sources_by_content.get(normalised_text, set()))
            if len(normalised_text) >= 500:
                candidates.update(self.sources_by_prefix.get(normalised_text[:500], set()))
            for source_id in candidates & self.known_sources:
                source_ranks[source_id] = min(rank, source_ranks.get(source_id, rank))

        direct_node_ids: set[str] = set()
        direct_edge_indexes: set[int] = set()
        for source_id in source_ranks:
            direct_node_ids.update(self.node_ids_by_source.get(source_id, set()))
            direct_edge_indexes.update(self.edge_indexes_by_source.get(source_id, set()))
        for edge_index in direct_edge_indexes:
            edge = self.edges[edge_index]
            direct_node_ids.update((edge.source, edge.target))

        def rank_sources(values: tuple[str, ...]) -> int:
            return min(
                (source_ranks[source] for source in values if source in source_ranks),
                default=10**9,
            )

        selected_nodes = sorted(
            (self.nodes[node_id] for node_id in direct_node_ids if node_id in self.nodes),
            key=lambda node: (rank_sources(node.source_ids), node.name.casefold(), node.node_id),
        )[: self.max_nodes]
        selected_node_ids = {node.node_id for node in selected_nodes}
        selected_edges = sorted(
            (
                self.edges[index]
                for index in direct_edge_indexes
                if self.edges[index].source in selected_node_ids
                and self.edges[index].target in selected_node_ids
            ),
            key=lambda edge: (
                rank_sources(edge.source_ids),
                edge.source.casefold(),
                edge.target.casefold(),
                edge.relation.casefold(),
            ),
        )[: self.max_edges]

        def refs(values: tuple[str, ...]) -> str:
            ranks = sorted({source_ranks[source] for source in values if source in source_ranks})
            return ",".join(f"E{rank}" for rank in ranks) or "unknown"

        sections: list[str] = []
        if selected_nodes:
            lines = [f"### {self.label}: Entities/events", "name | type | description | source"]
            for node in selected_nodes:
                lines.append(
                    " | ".join(
                        (
                            _clean(node.name),
                            _clean(node.node_type),
                            _clean(node.description),
                            refs(node.source_ids),
                        )
                    )
                )
            sections.append("\n".join(lines))
        if selected_edges:
            lines = [
                f"### {self.label}: Relationships",
                "source | relationship | target | description | evidence",
            ]
            for edge in selected_edges:
                lines.append(
                    " | ".join(
                        (
                            _clean(self.nodes[edge.source].name),
                            _clean(edge.relation),
                            _clean(self.nodes[edge.target].name),
                            _clean(edge.description),
                            refs(edge.source_ids),
                        )
                    )
                )
            sections.append("\n".join(lines))
        prefix = re.sub(r"[^a-z0-9]+", "_", self.label.casefold()).strip("_") or "graph"
        return sections, {
            f"{prefix}_entities_events": len(selected_nodes),
            f"{prefix}_relationships": len(selected_edges),
            f"{prefix}_matched_sources": len(source_ranks),
        }

    def render(
        self,
        row: dict[str, Any],
        *,
        top_k: int,
    ) -> tuple[str, dict[str, int]]:
        sections, stats = self.render_sections(row, top_k=top_k)
        return "\n\n".join(sections), stats


class NativeContextRegistry:
    def __init__(self) -> None:
        self.views: dict[str, list[SourceAlignedNativeGraph]] = defaultdict(list)

    def add(self, method: str, view: SourceAlignedNativeGraph) -> None:
        self.views[method].append(view)

    def render(
        self,
        method: str,
        row: dict[str, Any],
        *,
        top_k: int,
    ) -> tuple[str, dict[str, int]]:
        sections: list[str] = []
        stats: dict[str, int] = {}
        for view in self.views.get(method, []):
            view_sections, view_stats = view.render_sections(row, top_k=top_k)
            sections.extend(view_sections)
            stats.update(view_stats)
        return "\n\n".join(sections), stats

    def render_sections(
        self,
        method: str,
        row: dict[str, Any],
        *,
        top_k: int,
    ) -> tuple[list[str], dict[str, int]]:
        sections: list[str] = []
        stats: dict[str, int] = {}
        for view in self.views.get(method, []):
            view_sections, view_stats = view.render_sections(row, top_k=top_k)
            sections.extend(view_sections)
            stats.update(view_stats)
        return sections, stats


def _display_node_name(node_id: str) -> str:
    # E2RAG adds a source suffix to per-document entity nodes.
    value = re.sub(r"_chunk-[0-9a-f]{20,64}$", "", node_id, flags=re.IGNORECASE)
    return value.strip('"')
