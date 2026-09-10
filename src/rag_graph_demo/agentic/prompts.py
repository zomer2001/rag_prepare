from __future__ import annotations

import hashlib

from rag_graph_demo.evaluation.models import RetrievedChunk

AGENTIC_PROMPT_PROFILE_VERSION = "agentic_retrieval_variants_v2"
EVONEED_PROMPT_PROFILE_VERSION = "ours_evoneed_v3_joint_plan_fixed_budget"
EVONEED_DEEP_PROMPT_PROFILE_VERSION = "ours_evoneed_deep_v1_legacy_plan_assess"
EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION = (
    "ours_evoneed_deep_oneplan_v1_direct_retrieval_plan"
)

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

EVONEED_PLAN_SYSTEM = """You are the knowledge-need planner for a training-free RAG
system. Decompose the question into evidence needs; do not answer it and do not add
unstated facts. Return one JSON object with analytic_goal, sufficient=false, needs,
established_findings=[], conflicts=[], and stop_reason="".

Each need has: need_id, knowledge_type, status="missing", query, route, budget, rationale,
low_level_keywords, high_level_keywords, entities, relations, event_types, start_time,
end_time, state_intents, expand_event_chain, missing_slots, and supported_evidence_ids=[].
Use only these mappings:
- text_evidence -> vector: exact wording, numbers, conditions, quotations, or source text.
- relation_evidence -> graph: entities, predicates, bridge relations, and multi-hop links.
- evolution_evidence -> event: how an entity or relation changes through time.

For evolution_evidence, missing_slots may use state_before, change_event,
effective_time, state_after, stance_status, and supporting_source. Produce at most
three non-overlapping needs. Needs represent evidence types, not individual sources:
group multiple source-text anchors into one text_evidence need when possible. If the
question asks whether a state, relation, focus, stance, or portrayal changed or
remained consistent across time, always include one evolution_evidence need. Every
query must preserve the named entities, dates, sources, and constraints relevant to
that need. low_level_keywords contain concrete names and identifiers;
high_level_keywords contain relations, actions, and themes. state_intents may contain
only latest, change, conflict, denial, correction, or replacement. Set
expand_event_chain=true only for temporal order, change, cause, correction, or
replacement questions. This single plan is executed directly: do not assume another
planner will reinterpret a need. Assign every need a positive integer budget and make
the budgets sum exactly to the Total initial retrieval budget supplied by the user.
Allocate more budget to the needs most important or difficult for answering the
question. Return JSON only."""

# Kept as a separate, frozen profile so the high-budget experiment can reproduce
# the successful 2026-09-04 planner without changing the controlled EvoNeed variants.
EVONEED_DEEP_PLAN_SYSTEM = """You are the knowledge-need planner for a training-free RAG
system. Decompose the question into evidence needs; do not answer it and do not add
unstated facts. Return one JSON object with analytic_goal, sufficient=false, needs,
established_findings=[], conflicts=[], and stop_reason="".

Each need has: need_id, knowledge_type, status="missing", query, route, rationale,
entities, relations, event_types, start_time, end_time, missing_slots, and
supported_evidence_ids=[]. Use only these mappings:
- text_evidence -> vector: exact wording, numbers, conditions, quotations, or source text.
- relation_evidence -> graph: entities, predicates, bridge relations, and multi-hop links.
- evolution_evidence -> event: how an entity or relation changes through time.

For evolution_evidence, missing_slots may use state_before, change_event,
effective_time, state_after, stance_status, and supporting_source. Produce at most
three non-overlapping needs. Needs represent evidence types, not individual sources:
group multiple source-text anchors into one text_evidence need when possible. If the
question asks whether a state, relation, focus, stance, or portrayal changed or
remained consistent across time, always include one evolution_evidence need. Every
query must preserve the named entities, dates, sources, and constraints relevant to
that need. Return JSON only."""

EVONEED_DEEP_ONEPLAN_SYSTEM = """You are the knowledge-need and retrieval planner for a
training-free RAG system. Decompose the question into evidence needs; do not answer it
and do not add unstated facts. Return one JSON object with analytic_goal,
sufficient=false, needs, established_findings=[], conflicts=[], and stop_reason="".

Each need has: need_id, knowledge_type, status="missing", query, route, rationale,
low_level_keywords, high_level_keywords, entities, relations, event_types, start_time,
end_time, state_intents, expand_event_chain, missing_slots, and
supported_evidence_ids=[]. Use only these mappings:
- text_evidence -> vector: exact wording, numbers, conditions, quotations, or source text.
- relation_evidence -> graph: entities, predicates, bridge relations, and multi-hop links.
- evolution_evidence -> event: how an entity or relation changes through time.

Produce at most three non-overlapping needs and allow multiple distinct needs to use
the same route. Needs represent evidence types, not individual sources. Every query
must preserve the named entities, dates, sources, and constraints relevant to that
need. low_level_keywords contain concrete names and identifiers; high_level_keywords
contain relations, actions, and themes. entities contain only participants or affected
objects explicitly named in the question; do not treat a reporting source as an event
entity unless the event happens to that source. Use start_time and end_time only when
the question explicitly supplies a date or year. state_intents may contain only latest,
change, conflict, denial, correction, or replacement. Set expand_event_chain=true only
for temporal order, change, cause, correction, or replacement questions.

For evolution_evidence, missing_slots may use state_before, change_event,
effective_time, state_after, stance_status, and supporting_source. If the question asks
whether a state, relation, focus, stance, or portrayal changed or remained consistent
across time, always include one evolution_evidence need. This plan is executed directly:
make every route-specific field complete because no second query planner will run.
Return JSON only."""

EVONEED_ASSESS_SYSTEM = """You are the evidence-gap controller for a training-free
text--relation--event RAG system. Judge only the supplied evidence. Update the
KnowledgeNeedState and return one JSON object with analytic_goal, sufficient, needs,
established_findings, conflicts, and stop_reason.

For every previous need, set status to satisfied only when cited evidence directly
supports it; otherwise use partial or missing. Put valid supplied evidence IDs in
supported_evidence_ids. For every remaining gap, write a focused next query and route:
text_evidence->vector, relation_evidence->graph, evolution_evidence->event. Evolution
needs must explicitly list any missing state_before, change_event, effective_time,
state_after, stance_status, or supporting_source slots. Add a new need only when the
current evidence exposes a necessary gap not represented previously. Use at most
three open needs and do not infer that a fact is absent merely because retrieval did
not find it. Every open need searched in the previous round must receive a materially
different next query: retain its source, entity, date, and relation anchors, then add
only the unresolved evidence slots or useful aliases. Never repeat the previous query
verbatim. Mark sufficient=true only when every part of the analytic goal is grounded.
Return JSON only."""


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


def evoneed_prompt_profile_digest() -> str:
    payload = "\x1f".join(
        (
            EVONEED_PROMPT_PROFILE_VERSION,
            EVONEED_PLAN_SYSTEM,
            EVONEED_ASSESS_SYSTEM,
            GROUNDED_ANSWER_SYSTEM,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def evoneed_deep_prompt_profile_digest() -> str:
    payload = "\x1f".join(
        (
            EVONEED_DEEP_PROMPT_PROFILE_VERSION,
            EVONEED_DEEP_PLAN_SYSTEM,
            EVONEED_ASSESS_SYSTEM,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def evoneed_deep_oneplan_prompt_profile_digest() -> str:
    payload = "\x1f".join(
        (
            EVONEED_DEEP_ONEPLAN_PROMPT_PROFILE_VERSION,
            EVONEED_DEEP_ONEPLAN_SYSTEM,
            EVONEED_ASSESS_SYSTEM,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()
