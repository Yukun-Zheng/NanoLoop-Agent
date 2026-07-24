"""Persistent, evidence-bearing multi-turn conversation contracts."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.contracts.common import ContractModel
from app.contracts.enums import QueryType
from app.contracts.queries import Citation, MaterialContext, ToolCallLog, ToolEvidence


class CreateConversationRequest(ContractModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)


class ConversationMessageRequest(ContractModel):
    content: str = Field(min_length=1, max_length=4_000)
    query_type: QueryType = QueryType.AUTO
    image_id: str | None = None
    run_ids: list[str] = Field(default_factory=list, max_length=50)
    material_context: MaterialContext | None = None


class ChatTurnEvidenceDTO(ContractModel):
    citations: list[Citation] = Field(default_factory=list)
    data_evidence: list[ToolEvidence] = Field(default_factory=list)
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    llm_provider: str
    llm_model: str | None = None
    fallback_used: bool = False
    generation_time_ms: int = Field(ge=0)
    prompt_template_id: str
    prompt_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChatMessageDTO(ContractModel):
    message_id: str
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    query_type: QueryType
    image_id: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    material_context: MaterialContext | None = None
    confidence: Literal["low", "medium", "high"] | None = None
    outcome_code: Literal["OK", "INSUFFICIENT_EVIDENCE"] | None = None
    evidence: ChatTurnEvidenceDTO | None = None
    created_at: datetime


class ConversationSummaryDTO(ContractModel):
    conversation_id: str
    job_id: str
    title: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(ge=0)


class ConversationListData(ContractModel):
    conversations: list[ConversationSummaryDTO]


class ConversationDetailDTO(ConversationSummaryDTO):
    messages: list[ChatMessageDTO] = Field(default_factory=list)
