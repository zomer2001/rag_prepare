from __future__ import annotations

import json
from typing import Any, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel

from rag_graph_demo.agentic.models import (
    EvidenceAssessment,
    GroundedAnswer,
    ModelCall,
    QueryRewrite,
)
from rag_graph_demo.agentic.prompts import (
    EVIDENCE_ASSESSMENT_SYSTEM,
    GROUNDED_ANSWER_SYSTEM,
    QUERY_REWRITE_SYSTEM,
    evidence_context,
)
from rag_graph_demo.config import Settings
from rag_graph_demo.evaluation.models import RetrievedChunk


class AgentModel(Protocol):
    async def assess(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        *,
        retrieval_round: int,
    ) -> ModelCall[EvidenceAssessment]: ...

    async def rewrite(
        self,
        question: str,
        current_query: str,
        evidence: list[RetrievedChunk],
        assessment: EvidenceAssessment,
    ) -> ModelCall[QueryRewrite]: ...

    async def answer(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        assessment: EvidenceAssessment | None,
    ) -> ModelCall[GroundedAnswer]: ...


class DeepSeekAgentModel:
    """Typed DeepSeek facade used only by the LangGraph controller."""

    def __init__(self, settings: Settings) -> None:
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for agentic retrieval")
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def assess(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        *,
        retrieval_round: int,
    ) -> ModelCall[EvidenceAssessment]:
        user = (
            f"Question: {question}\nRetrieval round: {retrieval_round}\n\n"
            f"Evidence:\n{evidence_context(evidence)}\n\nReturn JSON only."
        )
        return await self._json_call(
            EvidenceAssessment,
            EVIDENCE_ASSESSMENT_SYSTEM,
            user,
            model=self.settings.deepseek_extraction_model,
        )

    async def rewrite(
        self,
        question: str,
        current_query: str,
        evidence: list[RetrievedChunk],
        assessment: EvidenceAssessment,
    ) -> ModelCall[QueryRewrite]:
        user = (
            f"Original question: {question}\nPrevious query: {current_query}\n"
            f"Missing information: {assessment.missing_information}\n"
            f"Assessment reason: {assessment.reason}\n"
            f"Evidence already found:\n{evidence_context(evidence, max_chars=8_000)}\n\n"
            "Return JSON only."
        )
        return await self._json_call(
            QueryRewrite,
            QUERY_REWRITE_SYSTEM,
            user,
            model=self.settings.deepseek_extraction_model,
        )

    async def answer(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        assessment: EvidenceAssessment | None,
    ) -> ModelCall[GroundedAnswer]:
        diagnosis = assessment.reason if assessment else "No assessment available"
        user = (
            f"Question: {question}\nEvidence diagnosis: {diagnosis}\n\n"
            f"Evidence:\n{evidence_context(evidence)}\n\nReturn JSON only."
        )
        return await self._json_call(
            GroundedAnswer,
            GROUNDED_ANSWER_SYSTEM,
            user,
            model=self.settings.deepseek_chat_model,
        )

    async def _json_call(
        self,
        schema: type[BaseModel],
        system: str,
        user: str,
        *,
        model: str,
    ) -> ModelCall[Any]:
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "extra_body": {
                "thinking": {
                    "type": (
                        "enabled" if self.settings.deepseek_thinking_enabled else "disabled"
                    )
                }
            },
        }
        if not self.settings.deepseek_thinking_enabled:
            request["temperature"] = 0
        response = await self.client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        value = schema.model_validate(extract_json_object(content))
        usage = response.usage
        return ModelCall(
            value=value,
            input_tokens=int(usage.prompt_tokens) if usage else 0,
            output_tokens=int(usage.completion_tokens) if usage else 0,
        )


def extract_json_object(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("DeepSeek did not return a JSON object") from None
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek JSON response must be an object")
    return parsed
