from __future__ import annotations

import re
from collections import defaultdict

from rag_graph_demo.retrieval.models import EvidenceBundle, EvidenceCandidate


def fuse_evidence(
    channel_candidates: dict[str, list[EvidenceCandidate]],
    *,
    top_k: int,
    rrf_k: int = 60,
    passages_per_document: int = 3,
    linked_candidate_weight: float = 0.1,
) -> list[EvidenceBundle]:
    """Fuse path ranks while treating cross-view expansion as one bounded bonus."""
    if not 0 <= linked_candidate_weight <= 1:
        raise ValueError("linked_candidate_weight must be between 0 and 1")
    scores: dict[str, float] = defaultdict(float)
    linked_scores: dict[str, float] = defaultdict(float)
    grouped: dict[str, list[EvidenceCandidate]] = defaultdict(list)
    channel_order = {"vector": 0, "graph": 1, "event": 2}
    for _channel, values in channel_candidates.items():
        seen_documents: set[str] = set()
        for candidate in values:
            if not candidate.document_id:
                continue
            grouped[candidate.document_id].append(candidate)
            if candidate.document_id in seen_documents:
                continue
            seen_documents.add(candidate.document_id)
            rank_score = 1.0 / (rrf_k + candidate.rank)
            if candidate.metadata.get("linked_by") == "ours_evolink":
                # Graph and event expansion are correlated views of one link. Count
                # only the strongest link once instead of awarding another full path.
                linked_scores[candidate.document_id] = max(
                    linked_scores[candidate.document_id],
                    rank_score,
                )
            else:
                scores[candidate.document_id] += rank_score

    for document_id, linked_score in linked_scores.items():
        scores[document_id] += linked_candidate_weight * linked_score

    ordered_documents = sorted(
        scores,
        key=lambda document_id: (scores[document_id], document_id),
        reverse=True,
    )[:top_k]
    bundles: list[EvidenceBundle] = []
    for document_id in ordered_documents:
        candidates = sorted(
            grouped[document_id],
            key=lambda item: (item.rank, channel_order[item.channel], item.evidence_id),
        )
        passages: list[EvidenceCandidate] = []
        seen_text: set[str] = set()
        seen_channels: set[str] = set()
        # First preserve the best independently traceable item from every path.
        for candidate in candidates:
            normalized = re.sub(r"\W+", "", candidate.text.casefold())
            if candidate.channel in seen_channels:
                continue
            passages.append(candidate)
            seen_channels.add(candidate.channel)
            seen_text.add(normalized)
            if len(passages) >= passages_per_document:
                break
        if len(passages) < passages_per_document:
            for candidate in candidates:
                normalized = re.sub(r"\W+", "", candidate.text.casefold())
                if normalized in seen_text:
                    continue
                passages.append(candidate)
                seen_text.add(normalized)
                if len(passages) >= passages_per_document:
                    break
        channels = tuple(
            channel
            for channel in ("vector", "graph", "event")
            if channel in {item.channel for item in grouped[document_id]}
        )
        bundles.append(
            EvidenceBundle(
                document_id=document_id,
                score=scores[document_id],
                channels=channels,  # type: ignore[arg-type]
                passages=tuple(passages),
            )
        )
    return bundles
