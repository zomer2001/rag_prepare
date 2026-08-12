from __future__ import annotations

import hashlib

from rag_graph_demo.evaluation.models import RetrievedChunk

AGENTIC_PROMPT_PROFILE_VERSION = "agentic_retrieval_variants_v2"

EVIDENCE_ASSESSMENT_SYSTEM = (
    "You are the evidence controller in an agentic RAG system. Judge only the supplied "
    "evidence. Return one JSON object with keys relevant (boolean), sufficient (boolean), "
    "reason (string), and missing_information (string). Evidence is sufficient only when "
    "it supports a complete, grounded answer to every part of the question."
)

QUERY_REWRITE_SYSTEM = (
    "You rewrite queries for an information retrieval system. Return one JSON object with keys "
    "query and rationale. Preserve named entities, dates, and constraints. Use the "
    "missing-information diagnosis to produce a focused query that is materially different "
    "from the previous query."
)

GROUNDED_ANSWER_SYSTEM = (
    "You answer strictly from supplied evidence. Return one JSON object with keys answer "
    "(string), citations (array of evidence_id strings), and insufficient_evidence (boolean). "
    "Every factual claim must be supported by cited evidence. If evidence is insufficient, "
    "say what cannot be established and set insufficient_evidence to true. Do not cite IDs "
    "that are absent from the evidence. Event ordering alone does not establish causality."
)


def evidence_id(item: RetrievedChunk) -> str:
    return item.document_id or item.chunk_id


def evidence_context(values: list[RetrievedChunk], *, max_chars: int = 30_000) -> str:
    blocks: list[str] = []
    used = 0
    for item in values:
        channels = ",".join(str(value) for value in item.metadata.get("channels") or [])
        block = (
            f"[evidence_id={evidence_id(item)}; document_id={item.document_id or 'unknown'}; "
            f"channels={channels or 'unknown'}]\n{item.text.strip()}"
        )
        remaining = max_chars - used
        if remaining <= 0:
            break
        blocks.append(block[:remaining])
        used += min(len(block), remaining)
    return "\n\n".join(blocks)


def prompt_profile_digest() -> str:
    payload = "\x1f".join(
        (
            AGENTIC_PROMPT_PROFILE_VERSION,
            EVIDENCE_ASSESSMENT_SYSTEM,
            QUERY_REWRITE_SYSTEM,
            GROUNDED_ANSWER_SYSTEM,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()
