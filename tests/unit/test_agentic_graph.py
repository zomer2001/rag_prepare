from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rag_graph_demo.agentic.graph import AgenticRAGGraph
from rag_graph_demo.agentic.model import DeepSeekAgentModel, extract_json_object
from rag_graph_demo.agentic.models import (
    AgenticRun,
    EvidenceAssessment,
    GroundedAnswer,
    ModelCall,
    QueryRewrite,
)
from rag_graph_demo.config import Settings
from rag_graph_demo.evaluation.models import (
    BenchmarkDataset,
    BenchmarkDocument,
    BenchmarkQuery,
    RetrievedChunk,
)
from rag_graph_demo.evaluation.runner import EvaluationRunner, agentic_diagnostics


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def __call__(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        assert top_k == 10
        if query == "focused event query":
            return [
                RetrievedChunk(
                    chunk_id="event-chunk",
                    document_id="doc-good",
                    text="Alpha deployed the system in March after the January contract.",
                    score=0.9,
                    metadata={"channels": ["vector", "graph", "event"]},
                )
            ]
        return [
            RetrievedChunk(
                chunk_id="noise-chunk",
                document_id="doc-noise",
                text="Unrelated retail earnings.",
                score=0.8,
                metadata={"channels": ["vector"]},
            )
        ]


class FakeModel:
    def __init__(self) -> None:
        self.assessment_count = 0

    async def assess(self, question, evidence, *, retrieval_round):
        self.assessment_count += 1
        sufficient = self.assessment_count == 2
        return ModelCall(
            EvidenceAssessment(
                relevant=sufficient,
                sufficient=sufficient,
                reason="enough" if sufficient else "missing event evidence",
                missing_information="deployment after contract" if not sufficient else "",
            ),
            input_tokens=10,
            output_tokens=2,
        )

    async def rewrite(self, question, current_query, evidence, assessment):
        return ModelCall(
            QueryRewrite(query="focused event query", rationale="target missing event"),
            input_tokens=5,
            output_tokens=2,
        )

    async def answer(self, question, evidence, assessment):
        return ModelCall(
            GroundedAnswer(
                answer="Alpha deployed the system in March.",
                citations=["doc-good", "invented-id"],
                insufficient_evidence=False,
            ),
            input_tokens=12,
            output_tokens=4,
        )


@pytest.mark.asyncio
async def test_agentic_graph_rewrites_and_reuses_three_path_retriever() -> None:
    retriever = FakeRetriever()
    graph = AgenticRAGGraph(retriever, FakeModel())

    result = await graph.run(
        "What happened after the contract?",
        top_k=10,
        context_k=10,
        max_retrieval_rounds=2,
        rrf_k=60,
    )

    assert retriever.queries == [
        "What happened after the contract?",
        "focused event query",
    ]
    assert result.retrieval_rounds == 2
    assert result.llm_calls == 4
    assert result.input_tokens == 37
    assert result.output_tokens == 10
    assert result.citations == ("doc-good",)
    assert {item.document_id for item in result.evidence} == {"doc-good", "doc-noise"}
    assert all(item.metadata["fusion"] == "agentic_round_rrf" for item in result.evidence)
    assert [item["node"] for item in result.trace] == [
        "retrieve",
        "assess",
        "rewrite",
        "retrieve",
        "assess",
        "answer",
    ]


def test_agentic_json_parser_accepts_fenced_output() -> None:
    assert extract_json_object('```json\n{"sufficient": true}\n```') == {
        "sufficient": True
    }


@pytest.mark.asyncio
async def test_agentic_json_call_retries_truncated_output(monkeypatch) -> None:
    invalid = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"relevant": true, "reason": "cut'),
                finish_reason="length",
            )
        ],
        usage=None,
    )
    valid = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"relevant": true, "sufficient": false, '
                        '"reason": "retry", "missing_information": "evidence"}'
                    )
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    create = AsyncMock(side_effect=[invalid, valid])
    model = DeepSeekAgentModel(
        Settings(_env_file=None, deepseek_api_key="test", llm_max_retries=2)
    )
    model.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(
        "rag_graph_demo.agentic.model.asyncio.sleep", AsyncMock()
    )

    call = await model._json_call(
        EvidenceAssessment,
        "Return JSON.",
        "Question?",
        model="test-model",
    )

    assert call.value.reason == "retry"
    assert create.await_count == 2


def _dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        name="agentic-fixture",
        documents=(BenchmarkDocument("doc-good", "Title", "Evidence."),),
        queries=(BenchmarkQuery("q-1", "Question?", "fact", "Answer", ()),),
    )


def test_agentic_method_has_separate_retriever_and_manifest_profile(tmp_path) -> None:
    runner = EvaluationRunner(
        _dataset(),
        Settings(_env_file=None),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=(
            "ours",
            "agentic_hybrid",
            "agentic_light_entity",
            "agentic_ours",
            "agentic_ours_tkg",
        ),
        show_progress=False,
    )

    assert runner._method_retriever("ours", None, None).__func__ is (
        runner._retrieve_ours.__func__
    )
    assert runner._method_retriever("agentic_ours", None, None).__func__ is (
        runner._retrieve_agentic_ours.__func__
    )
    assert runner._method_retriever("agentic_ours_tkg", None, None).__func__ is (
        runner._retrieve_agentic_ours_tkg.__func__
    )
    assert runner._method_retriever("agentic_hybrid", None, None).__func__ is (
        runner._retrieve_agentic_hybrid.__func__
    )
    assert runner._method_retriever("agentic_light_entity", None, None).__func__ is (
        runner._retrieve_agentic_light_entity.__func__
    )
    runner._ensure_output_manifest()
    manifest = json.loads((tmp_path / "output" / "run_manifest.json").read_text())

    assert manifest["agentic_profile"]["version"] == "agentic_retrieval_variants_v2"
    assert manifest["agentic_profile"]["max_retrieval_rounds"] == 2
    assert manifest["agentic_profile"]["methods"] == [
        "agentic_hybrid",
        "agentic_light_entity",
        "agentic_ours",
        "agentic_ours_tkg",
    ]


@pytest.mark.asyncio
async def test_evaluation_runner_checkpoints_agentic_trajectory(tmp_path) -> None:
    runner = EvaluationRunner(
        _dataset(),
        Settings(_env_file=None),
        runtime_dir=tmp_path / "runtime",
        output_dir=tmp_path / "output",
        methods=("agentic_ours",),
        show_progress=False,
    )
    evidence = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-good",
        text="Evidence.",
        metadata={"channels": ["vector", "graph", "event"]},
    )

    async def retrieve(_question: str, _top_k: int) -> AgenticRun:
        return AgenticRun(
            question="Question?",
            answer="Answer",
            citations=("doc-good",),
            evidence=(evidence,),
            trace=({"node": "answer"},),
            retrieval_rounds=1,
            llm_calls=2,
            input_tokens=20,
            output_tokens=5,
            insufficient_evidence=False,
        )

    rows = await runner._evaluate_method("agentic_ours", retrieve)
    saved = json.loads((tmp_path / "output" / "agentic_ours.jsonl").read_text())

    assert rows[0]["generated_answer"] == "Answer"
    assert saved["trajectory"]["retrieval_rounds"] == 1
    assert saved["citations"] == ["doc-good"]
    assert agentic_diagnostics(rows)["mean_llm_calls"] == 2
