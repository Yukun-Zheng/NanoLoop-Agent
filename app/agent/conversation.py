"""Multi-turn orchestration over deterministic tools, RAG, and one LLM call."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import monotonic
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.agent.evidence_validation import validate_conversation_answer
from app.agent.router import QueryRouter
from app.agent.unified_query import DataQuery, DataQueryResult, DataToolService
from app.analysis.authorization import require_read
from app.contracts.common import utc_now
from app.contracts.conversations import (
    ChatMessageDTO,
    ChatTurnEvidenceDTO,
    ConversationDetailDTO,
    ConversationListData,
    ConversationMessageRequest,
    ConversationSummaryDTO,
    CreateConversationRequest,
)
from app.contracts.enums import QueryType
from app.contracts.identity import AuthMode, PrincipalContext
from app.contracts.knowledge import RetrievedChunk
from app.contracts.queries import Citation, MaterialContext, ToolCallLog, ToolEvidence
from app.core.errors import ResourceNotFoundError, ServiceUnavailableError
from app.db.models import (
    AnalysisJob,
    ChatConversation,
    ChatMessage,
    ChatTurnEvidence,
    ImageAsset,
    SegmentationRun,
)
from app.db.repositories import SqlAlchemyRepositorySet
from app.rag.prompts import CHAT_PROMPT_TEMPLATE_ID, CHAT_PROMPT_TEMPLATE_SHA256
from app.rag.providers import (
    AnswerProviderError,
    CitationContext,
    CitationValidationError,
    ExtractiveAnswerProvider,
    OpenAICompatibleProvider,
    strip_thinking,
    validate_provider_answer,
)
from app.rag.service import KnowledgeEvidence, KnowledgeService
from app.research import OnlineResearchService, ResearchEvidence

_UNTRUSTED_REQUEST = re.compile(
    r"忽略(?:文献|引用|规则)|编造|虚构|ignore (?:the )?(?:rules|citations)|fabricate",
    re.I,
)
_NUMERIC_SENTENCE = re.compile(
    r".*?(?:[。！？；;\n]+|(?<!\d)[.!?]+(?!\d))|.+$",
    re.S,
)
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
_EVIDENCE_REFERENCE = re.compile(r"\[[DC]\d+\]")
_DATA_REFERENCE = re.compile(r"\[(D\d+)\]")
_MISSING_SCALE_CLAIM = re.compile(
    r"(?:缺乏|缺少).{0,8}(?:比例尺|物理尺度)|未提供.{0,6}比例尺|"
    r"(?:像素单位|像素尺度).{0,12}(?:推断|转换为真实)"
)
_NEGATED_MISSING_SCALE_CLAIM = re.compile(
    r"(?:不得|不应|不能).{0,8}(?:缺乏|缺少|未提供).{0,8}(?:比例尺|物理尺度)"
)
_CHEMICAL_SYMBOL_TEXT = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
    "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba "
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds "
    "Rg Cn Nh Fl Mc Lv Ts Og"
)
_CHEMICAL_SYMBOLS = frozenset(_CHEMICAL_SYMBOL_TEXT.split())
_CANDIDATE_SYSTEM_NOTES: Mapping[str, tuple[str, ...]] = {
    "BaCr": (
        "BaCr 只是 Ba–Cr 元素体系标签，不是可据此确认的电中性化学式。",
        "若实际物相为 BaCrO4（铬酸钡），电荷守恒要求 Cr 的形式氧化态为 +6；"
        "不得把 BaCrO4 描述为含 Cr(III)。",
        "BaCrO4 通常是黄色、难溶的无机铬酸盐，历史上用于颜料或防腐蚀体系；"
        "其中 Cr(VI) 具有氧化性和显著健康危害，讨论应用时必须同时提示安全风险。",
        "若实际为其他 Ba–Cr 氧化物、混合相或复合物，导电、磁性、热稳定、"
        "氧化还原和催化行为都可能改变，不能从样品标签或 SEM 形貌确定方向。",
        "EDS 可确认元素组成与空间分布，XRD 用于物相识别，XPS 用于区分 Cr 价态；"
        "SEM 分割只能支撑粒径、覆盖、团聚和空间分布等形貌结论。",
    ),
}
_CANDIDATE_SYSTEM_ANSWERS: Mapping[str, str] = {
    "BaCr": (
        "`BaCr-1` 可以安全识别为 Ba–Cr 候选体系的样品标签，但它不是已经确认的化学式。\n\n"
        "可能的性质要分物相讨论：\n"
        "- 若实际物相是 BaCrO₄（铬酸钡），它通常呈黄色、难溶；其中 Cr 为 +6 价，"
        "具有氧化性和显著健康危害，历史上曾用于颜料或防腐蚀体系。\n"
        "- 若实际是其他 Ba–Cr 氧化物、混合相或复合物，导电、磁性、热稳定、氧化还原"
        "和催化行为都可能随 Cr 价态与晶相改变，不能仅凭标签判断增强还是减弱。\n"
        "- 当前 SEM 图像与分割结果能回答颗粒尺寸、覆盖、团聚和空间分布，不能直接证明"
        "元素组成、晶相、价态或功能性能。\n\n"
        "最有效的确认顺序是：先用 EDS 核对 Ba、Cr 及其他元素，再用 XRD 确认物相，"
        "最后用 XPS 区分 Cr 价态。在这些结果出来前，不应把样品直接定名为 BaCrO₄，"
        "也不应宣称它已经具有某种催化、导电或磁学性能。"
    ),
}
_SAMPLE_LABEL_STEM = re.compile(r"^[A-Z][A-Za-z]{1,11}$")
_ELEMENT_SYMBOL = re.compile(r"[A-Z][a-z]?")
logger = logging.getLogger(__name__)


class ConversationService:
    """Persist bounded history and synthesize one validated answer per turn."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        router: QueryRouter,
        data_tools: DataToolService,
        knowledge_service: KnowledgeService,
        research_service: OnlineResearchService | None = None,
        llm_provider: OpenAICompatibleProvider | None,
        history_turns: int,
        history_max_chars: int,
    ) -> None:
        self.session_factory = session_factory
        self.router = router
        self.data_tools = data_tools
        self.knowledge_service = knowledge_service
        self.research_service = research_service
        self.llm_provider = llm_provider
        self.history_turns = history_turns
        self.history_max_chars = history_max_chars
        self.extractive = ExtractiveAnswerProvider()

    def create(
        self,
        job_id: str,
        request: CreateConversationRequest,
        *,
        principal: PrincipalContext,
    ) -> ConversationDetailDTO:
        tenant_id = _tenant_id(principal)
        now = utc_now()
        conversation = ChatConversation(
            conversation_id=f"conv_{uuid4().hex}",
            tenant_id=tenant_id,
            job_id=job_id,
            title=request.title or "新对话",
            created_by=_principal_id(principal),
            created_at=now,
            updated_at=now,
        )
        session = self.session_factory()
        try:
            self._require_job(session, job_id, principal)
            session.add(conversation)
            session.commit()
            return _conversation_detail(conversation, [])
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list(
        self,
        job_id: str,
        *,
        principal: PrincipalContext,
    ) -> ConversationListData:
        tenant_id = _tenant_id(principal)
        session = self.session_factory()
        try:
            self._require_job(session, job_id, principal)
            rows = session.execute(
                select(ChatConversation, func.count(ChatMessage.message_id))
                .outerjoin(
                    ChatMessage,
                    ChatMessage.conversation_id == ChatConversation.conversation_id,
                )
                .where(
                    ChatConversation.job_id == job_id,
                    ChatConversation.tenant_id == tenant_id,
                )
                .group_by(ChatConversation.conversation_id)
                .order_by(ChatConversation.updated_at.desc())
            ).all()
            return ConversationListData(
                conversations=[
                    _conversation_summary(conversation, int(message_count))
                    for conversation, message_count in rows
                ]
            )
        finally:
            session.close()

    def get(
        self,
        job_id: str,
        conversation_id: str,
        *,
        principal: PrincipalContext,
    ) -> ConversationDetailDTO:
        session = self.session_factory()
        try:
            self._require_job(session, job_id, principal)
            conversation = self._conversation(
                session,
                job_id,
                conversation_id,
                tenant_id=_tenant_id(principal),
                with_messages=True,
            )
            return _conversation_detail(conversation, conversation.messages)
        finally:
            session.close()

    def send(
        self,
        job_id: str,
        conversation_id: str,
        request: ConversationMessageRequest,
        *,
        principal: PrincipalContext,
    ) -> ConversationDetailDTO:
        tenant_id = _tenant_id(principal)
        session = self.session_factory()
        try:
            self._require_job(session, job_id, principal)
            conversation = self._conversation(
                session,
                job_id,
                conversation_id,
                tenant_id=tenant_id,
                with_messages=True,
            )
            resolved = self._validated_request(session, job_id, request, tenant_id=tenant_id)
            task_context = self._task_context(
                session,
                job_id,
                resolved,
                tenant_id=tenant_id,
            )
            history = _bounded_history(
                conversation.messages,
                turns=self.history_turns,
                max_chars=self.history_max_chars,
            )
            previous_type = next(
                (
                    QueryType(message.query_type)
                    for message in reversed(conversation.messages)
                    if message.role == "assistant"
                ),
                None,
            )
            previous_user_question = next(
                (
                    message.content
                    for message in reversed(conversation.messages)
                    if message.role == "user" and message.content.strip()
                ),
                None,
            )
            query_type = resolved.query_type
            if query_type in {QueryType.AUTO, QueryType.GENERAL_CHAT}:
                decision = self.router.classify(
                    resolved.content,
                    material_context=resolved.material_context,
                    previous_query_type=previous_type,
                )
                if query_type is QueryType.AUTO or decision.query_type is not QueryType.AUTO:
                    query_type = decision.query_type
            if query_type is QueryType.AUTO:
                query_type = QueryType.GENERAL_CHAT
            if principal.auth_mode is AuthMode.PRINCIPAL and query_type in {
                QueryType.MATERIAL_KNOWLEDGE,
                QueryType.MIXED,
            }:
                raise ServiceUnavailableError(details={"component": "knowledge_tenant_scope"})

            now = utc_now()
            user_message = ChatMessage(
                message_id=f"msg_{uuid4().hex}",
                conversation_id=conversation_id,
                role="user",
                content=resolved.content,
                query_type=query_type.value,
                image_id=resolved.image_id,
                run_ids_json=list(resolved.run_ids),
                material_context_json=(
                    resolved.material_context.model_dump(mode="json")
                    if resolved.material_context
                    else None
                ),
                created_at=now,
            )
            session.add(user_message)
            if conversation.title == "新对话":
                conversation.title = _title_from_message(resolved.content)
            conversation.updated_at = now
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        answer = self._answer(
            job_id,
            resolved,
            query_type=query_type,
            tenant_id=tenant_id,
            history=history,
            previous_user_question=previous_user_question,
            task_context=task_context,
        )
        session = self.session_factory()
        try:
            conversation = self._conversation(
                session,
                job_id,
                conversation_id,
                tenant_id=tenant_id,
                with_messages=False,
            )
            assistant = ChatMessage(
                message_id=f"msg_{uuid4().hex}",
                conversation_id=conversation_id,
                role="assistant",
                content=strip_thinking(answer.content),
                query_type=query_type.value,
                image_id=resolved.image_id,
                run_ids_json=list(resolved.run_ids),
                material_context_json=(
                    resolved.material_context.model_dump(mode="json")
                    if resolved.material_context
                    else None
                ),
                confidence=answer.confidence,
                outcome_code=answer.outcome_code,
                created_at=utc_now(),
            )
            assistant.evidence = ChatTurnEvidence(
                message_id=assistant.message_id,
                citations_json=[item.model_dump(mode="json") for item in answer.citations],
                data_evidence_json=[item.model_dump(mode="json") for item in answer.data.evidence],
                tool_calls_json=[item.model_dump(mode="json") for item in answer.tool_calls],
                limitations_json=list(answer.limitations),
                llm_provider=answer.llm_provider,
                llm_model=answer.llm_model,
                fallback_used=answer.fallback_used,
                generation_time_ms=answer.generation_time_ms,
                prompt_template_id=CHAT_PROMPT_TEMPLATE_ID,
                prompt_template_sha256=CHAT_PROMPT_TEMPLATE_SHA256,
            )
            session.add(assistant)
            conversation.updated_at = assistant.created_at
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        return self.get(job_id, conversation_id, principal=principal)

    def _answer(
        self,
        job_id: str,
        request: ConversationMessageRequest,
        *,
        query_type: QueryType,
        tenant_id: str,
        history: Sequence[Mapping[str, str]],
        previous_user_question: str | None,
        task_context: Mapping[str, object],
    ) -> _TurnAnswer:
        started = monotonic()
        if _UNTRUSTED_REQUEST.search(request.content):
            return _policy_refusal(started)
        data = DataQueryResult(answer="")
        knowledge = KnowledgeEvidence((), (), (), "OK")
        research = ResearchEvidence()
        checked_candidate_answer = _checked_candidate_answer(
            request.content,
            task_context,
        )
        if query_type in {QueryType.ANALYSIS_DATA, QueryType.MIXED}:
            data = self.data_tools.answer(
                DataQuery(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    question=_data_tool_question(
                        request.content,
                        previous_user_question=previous_user_question,
                    ),
                    image_id=request.image_id,
                    run_ids=tuple(request.run_ids),
                )
            )
            if query_type is QueryType.ANALYSIS_DATA and data.evidence:
                data = replace(
                    data,
                    evidence=(
                        *data.evidence,
                        _analysis_guardrail_evidence(data.evidence),
                    ),
                )
        if query_type in {QueryType.MATERIAL_KNOWLEDGE, QueryType.MIXED}:
            knowledge = self.knowledge_service.collect_evidence(
                request.content,
                material_context=request.material_context,
            )
        if knowledge.blocked_untrusted_instruction:
            return _policy_refusal(started)
        if (
            query_type in {QueryType.MATERIAL_KNOWLEDGE, QueryType.MIXED}
            and self.research_service is not None
        ):
            research = self.research_service.collect(request.content)
            knowledge = _merge_research_evidence(knowledge, research)

        provider = self.llm_provider
        if provider is not None and provider.health().status != "unavailable":
            try:
                generated = provider.generate_conversation(
                    question=request.content,
                    query_type=query_type.value,
                    history=history,
                    data_evidence=data.evidence,
                    contexts=knowledge.contexts,
                    material_context=request.material_context,
                    task_context=task_context,
                )
                answer = generated.answer
                provider_limitations = generated.limitations
                confidence = generated.confidence
                scientific_guardrail_applied = checked_candidate_answer is not None
                if checked_candidate_answer is not None:
                    answer = checked_candidate_answer
                    provider_limitations = (
                        "样品标签只提供候选元素体系，最终组成与性质仍需成分、物相和价态表征",
                    )
                    confidence = "medium"
                validation_limitations = provider_limitations
                validation_data_ids = generated.used_data_ids
                validation_citation_ids = generated.used_citation_ids
                if query_type is QueryType.GENERAL_CHAT:
                    answer = _EVIDENCE_REFERENCE.sub("", answer).strip()
                    provider_limitations = tuple(
                        cleaned
                        for item in provider_limitations
                        if (cleaned := _EVIDENCE_REFERENCE.sub("", item).strip())
                    )
                    validation_limitations = provider_limitations
                    validation_data_ids = ()
                    validation_citation_ids = ()
                if query_type is QueryType.ANALYSIS_DATA:
                    _validate_qualitative_analysis_synthesis(
                        answer=answer,
                        physical_scale_available=_guardrail_has_physical_scale(data.evidence),
                    )
                    validation_limitations = ()
                    validation_data_ids = tuple(
                        sorted(
                            set(_DATA_REFERENCE.findall(answer)),
                            key=lambda item: int(item[1:]),
                        )
                    )
                validate_conversation_answer(
                    answer=answer,
                    limitations=validation_limitations,
                    used_data_ids=validation_data_ids,
                    used_citation_ids=validation_citation_ids,
                    data_evidence=data.evidence,
                    citation_contexts=knowledge.contexts,
                    allow_uncited_general_chat=query_type is QueryType.GENERAL_CHAT,
                )
                used_citations = set(validation_citation_ids)
                content = answer
                if query_type is QueryType.ANALYSIS_DATA:
                    numeric_evidence_count = sum(
                        item.tool_name != "interpret_analysis_guardrails" for item in data.evidence
                    )
                    deterministic = _cite_deterministic_data(
                        data.answer,
                        numeric_evidence_count,
                    )
                    if deterministic:
                        content = f"{deterministic}\n\n本地模型综合：{answer}"
                generated_limitations = (
                    _deterministic_analysis_limitations(data.evidence)
                    if query_type is QueryType.ANALYSIS_DATA
                    else provider_limitations
                )
                return _TurnAnswer(
                    content=content,
                    data=data,
                    tool_calls=tuple((*data.tool_calls, *research.tool_calls)),
                    citations=tuple(
                        citation
                        for citation in knowledge.citations
                        if citation.citation_id in used_citations
                    ),
                    limitations=tuple(
                        dict.fromkeys(
                            [*data.limitations, *knowledge.limitations, *generated_limitations]
                        )
                    ),
                    confidence=confidence,
                    outcome_code=_outcome(query_type, data, knowledge),
                    llm_provider=(
                        "openai_compatible+scientific_guardrail"
                        if scientific_guardrail_applied
                        else "openai_compatible"
                    ),
                    llm_model=provider.model,
                    fallback_used=False,
                    generation_time_ms=_elapsed_ms(started),
                )
            except (AnswerProviderError, CitationValidationError) as error:
                logger.info(
                    "conversation_llm_fallback",
                    extra={
                        "reason": type(error).__name__,
                        "detail": str(error),
                        "query_type": query_type.value,
                    },
                )
        return self._fallback(
            request,
            query_type=query_type,
            data=data,
            knowledge=knowledge,
            research_tool_calls=research.tool_calls,
            generation_time_ms=_elapsed_ms(started),
        )

    def _fallback(
        self,
        request: ConversationMessageRequest,
        *,
        query_type: QueryType,
        data: DataQueryResult,
        knowledge: KnowledgeEvidence,
        research_tool_calls: Sequence[ToolCallLog],
        generation_time_ms: int,
    ) -> _TurnAnswer:
        data_answer = _cite_deterministic_data(data.answer, len(data.evidence))
        knowledge_answer = ""
        citations: tuple[Citation, ...] = ()
        knowledge_limitations: Sequence[str] = ()
        if knowledge.contexts:
            try:
                extracted = self.extractive.generate(
                    question=request.content,
                    contexts=knowledge.contexts,
                    material_context=request.material_context,
                )
                validate_provider_answer(
                    extracted,
                    {context.citation_id for context in knowledge.contexts},
                )
                knowledge_answer = extracted.answer
                knowledge_limitations = extracted.limitations
                used = set(extracted.used_citation_ids)
                citations = tuple(item for item in knowledge.citations if item.citation_id in used)
            except (AnswerProviderError, CitationValidationError):
                knowledge_answer = "知识库证据不足，无法基于当前已导入文档回答该问题。"
        if query_type is QueryType.GENERAL_CHAT:
            provider_available = (
                self.llm_provider is not None
                and self.llm_provider.health().status != "unavailable"
            )
            content = (
                "本地 Qwen 已连接，但本轮回答未通过证据校验，请直接重试或换一种问法。"
                if provider_available
                else "本地 Qwen 当前未连接。请启动或恢复 Ollama 后重试；"
                "实验数据和知识库查询仍可使用安全降级。"
            )
        elif query_type is QueryType.ANALYSIS_DATA:
            content = data_answer or "当前任务没有足够的已完成运行数据可供回答。"
        elif query_type is QueryType.MATERIAL_KNOWLEDGE:
            content = knowledge_answer or "知识库证据不足，无法确认。"
        else:
            content = (
                f"实验数据结论：\n{data_answer or '当前数据证据不足。'}\n\n"
                f"材料知识结论：\n{knowledge_answer or '当前知识证据不足。'}"
            )
        return _TurnAnswer(
            content=content,
            data=data,
            tool_calls=tuple((*data.tool_calls, *research_tool_calls)),
            citations=citations,
            limitations=tuple(
                dict.fromkeys(
                    [
                        *data.limitations,
                        *knowledge.limitations,
                        *knowledge_limitations,
                        "生成式回答不可用，已使用可信降级结果",
                    ]
                )
            ),
            confidence="low" if not (data.evidence or citations) else "medium",
            outcome_code=_outcome(query_type, data, knowledge),
            llm_provider="extractive",
            llm_model=None,
            fallback_used=True,
            generation_time_ms=generation_time_ms,
        )

    def _require_job(
        self,
        session: Session,
        job_id: str,
        principal: PrincipalContext,
    ) -> None:
        scope = SqlAlchemyRepositorySet(session).jobs.get_scope(
            job_id,
            tenant_id=_tenant_id(principal),
        )
        require_read(principal, scope)

    @staticmethod
    def _conversation(
        session: Session,
        job_id: str,
        conversation_id: str,
        *,
        tenant_id: str,
        with_messages: bool,
    ) -> ChatConversation:
        statement = select(ChatConversation).where(
            ChatConversation.conversation_id == conversation_id,
            ChatConversation.job_id == job_id,
            ChatConversation.tenant_id == tenant_id,
        )
        if with_messages:
            statement = statement.options(
                selectinload(ChatConversation.messages).selectinload(ChatMessage.evidence)
            )
        conversation = session.scalar(statement)
        if conversation is None:
            raise ResourceNotFoundError(
                details={
                    "resource": "conversation",
                    "job_id": job_id,
                    "conversation_id": conversation_id,
                }
            )
        return conversation

    @staticmethod
    def _task_context(
        session: Session,
        job_id: str,
        request: ConversationMessageRequest,
        *,
        tenant_id: str,
    ) -> Mapping[str, object]:
        job = session.scalar(
            select(AnalysisJob).where(
                AnalysisJob.job_id == job_id,
                AnalysisJob.tenant_id == tenant_id,
            )
        )
        image = (
            session.scalar(
                select(ImageAsset).where(
                    ImageAsset.image_id == request.image_id,
                    ImageAsset.job_id == job_id,
                )
            )
            if request.image_id
            else None
        )
        run_status_rows = session.execute(
            select(SegmentationRun.status, func.count(SegmentationRun.run_id))
            .where(SegmentationRun.job_id == job_id)
            .group_by(SegmentationRun.status)
        ).all()
        run_status_counts = {
            str(status): int(count) for status, count in run_status_rows
        }
        selected_run_rows = (
            session.scalars(
                select(SegmentationRun).where(
                    SegmentationRun.job_id == job_id,
                    SegmentationRun.run_id.in_(request.run_ids),
                )
            ).all()
            if request.run_ids
            else []
        )
        selected_runs_by_id = {run.run_id: run for run in selected_run_rows}
        selected_runs = [
            {
                "run_id": run.run_id,
                "model_id": run.model_id,
                "status": run.status,
                "roi_mode": run.roi_mode,
                "image_id": run.image_id,
            }
            for run_id in request.run_ids
            if (run := selected_runs_by_id.get(run_id)) is not None
        ]
        run_total = sum(run_status_counts.values())
        selected_run_configs = (
            session.scalars(
                select(SegmentationRun.run_config_json).where(
                    SegmentationRun.job_id == job_id,
                    SegmentationRun.run_id.in_(request.run_ids),
                )
            ).all()
            if request.run_ids
            else []
        )
        selected_run_has_physical_scale = any(
            isinstance(config, Mapping) and config.get("scale_nm_per_pixel") is not None
            for config in selected_run_configs
        )
        active_statuses = {
            "CREATED",
            "VALIDATING",
            "QUEUED",
            "PREPROCESSING",
            "SEGMENTING",
            "POSTPROCESSING",
            "QUALITY_CHECKING",
            "ANALYZING",
            "AGGREGATING",
        }
        completed_statuses = {"COMPLETED", "COMPLETED_WITH_WARNINGS"}
        if any(status in active_statuses for status in run_status_counts):
            next_steps = ["前往“运行进度”查看当前分析运行。"]
        elif any(status in completed_statuses for status in run_status_counts):
            next_steps = ["前往“查看结果”检查叠加图、统计和质量限制。"]
        else:
            next_steps = [
                "进入“开始分析”选择模型并创建一次全图分割运行。",
                "局部区域（ROI）是可选步骤，可以直接跳过。",
            ]
        candidate_material_hint = (
            _candidate_material_hint(image.sample_id, image.filename)
            if image is not None
            and not (image.material_name or image.material_formula)
            else None
        )
        return {
            "job": {
                "name": job.name if job is not None else None,
                "status": job.status if job is not None else None,
            },
            "selected_image": (
                {
                    "filename": image.filename,
                    "sample_id": image.sample_id,
                    "width_px": image.width,
                    "height_px": image.height,
                    "material_name": image.material_name,
                    "material_formula": image.material_formula,
                    "candidate_material_hint": candidate_material_hint,
                    "has_physical_scale": (
                        image.scale_nm_per_pixel is not None or selected_run_has_physical_scale
                    ),
                    "scale_nm_per_pixel": image.scale_nm_per_pixel,
                    "scale_source": image.scale_source,
                    "sem_metadata": image.sem_metadata_json or None,
                }
                if image is not None
                else None
            ),
            "runs": {
                "selected_count": len(request.run_ids),
                "selected": selected_runs,
                "selected_has_physical_scale": selected_run_has_physical_scale,
                "job_total_count": run_total,
                "status_counts": run_status_counts,
            },
            "available_next_steps": next_steps,
        }

    @staticmethod
    def _validated_request(
        session: Session,
        job_id: str,
        request: ConversationMessageRequest,
        *,
        tenant_id: str,
    ) -> ConversationMessageRequest:
        image: ImageAsset | None = None
        if request.image_id is not None:
            image = session.scalar(
                select(ImageAsset)
                .join(AnalysisJob, AnalysisJob.job_id == ImageAsset.job_id)
                .where(
                    ImageAsset.image_id == request.image_id,
                    ImageAsset.job_id == job_id,
                    AnalysisJob.tenant_id == tenant_id,
                )
            )
            if image is None:
                raise ResourceNotFoundError(
                    details={"resource": "image", "image_id": request.image_id}
                )
        run_ids = list(dict.fromkeys(request.run_ids))
        if run_ids:
            owned = set(
                session.scalars(
                    select(SegmentationRun.run_id)
                    .join(AnalysisJob, AnalysisJob.job_id == SegmentationRun.job_id)
                    .where(
                        SegmentationRun.job_id == job_id,
                        SegmentationRun.run_id.in_(run_ids),
                        AnalysisJob.tenant_id == tenant_id,
                    )
                ).all()
            )
            missing = [run_id for run_id in run_ids if run_id not in owned]
            if missing:
                raise ResourceNotFoundError(details={"resource": "run", "run_ids": missing})
        material = request.material_context
        if material is not None and material.source == "image_metadata":
            material = material.model_copy(update={"source": "request"})
        if (
            material is None
            and image is not None
            and (image.material_name or image.material_formula)
        ):
            material = MaterialContext(
                name=image.material_name,
                formula=image.material_formula,
                aliases=[],
                source="image_metadata",
            )
        return request.model_copy(
            update={
                "content": request.content.strip(),
                "run_ids": run_ids,
                "material_context": material,
            }
        )


def _candidate_material_hint(
    sample_id: str | None,
    filename: str | None,
) -> Mapping[str, object] | None:
    """Parse an element-system hint from a sample label without claiming composition."""

    label = (sample_id or "").strip()
    if not label and filename:
        label = filename.rsplit(".", 1)[0].strip()
    if not label:
        return None
    stem = re.split(r"[-_]", label, maxsplit=1)[0]
    if _SAMPLE_LABEL_STEM.fullmatch(stem) is None:
        return None
    symbols = _ELEMENT_SYMBOL.findall(stem)
    if len(symbols) < 2 or "".join(symbols) != stem:
        return None
    if any(symbol not in _CHEMICAL_SYMBOLS for symbol in symbols):
        return None
    hint: dict[str, object] = {
        "sample_label": label,
        "candidate_formula_stem": stem,
        "candidate_elements": symbols,
        "candidate_system": "–".join(symbols),
        "evidence_level": "unverified_label_hint",
    }
    if notes := _CANDIDATE_SYSTEM_NOTES.get(stem):
        hint["trusted_scientific_notes"] = list(notes)
    return hint


def _checked_candidate_answer(
    question: str,
    task_context: Mapping[str, object],
) -> str | None:
    normalized = question.casefold()
    if not any(
        signal in normalized
        for signal in ("性质", "特性", "性能", "用途", "应用", "什么材料")
    ):
        return None
    selected_image = task_context.get("selected_image")
    if not isinstance(selected_image, Mapping):
        return None
    hint = selected_image.get("candidate_material_hint")
    if not isinstance(hint, Mapping) or not hint.get("trusted_scientific_notes"):
        return None
    stem = hint.get("candidate_formula_stem")
    return _CANDIDATE_SYSTEM_ANSWERS.get(str(stem))


class _TurnAnswer:
    def __init__(
        self,
        *,
        content: str,
        data: DataQueryResult,
        tool_calls: tuple[ToolCallLog, ...],
        citations: tuple[Citation, ...],
        limitations: tuple[str, ...],
        confidence: str,
        outcome_code: str,
        llm_provider: str,
        llm_model: str | None,
        fallback_used: bool,
        generation_time_ms: int,
    ) -> None:
        self.content = content
        self.data = data
        self.tool_calls = tool_calls
        self.citations = citations
        self.limitations = limitations
        self.confidence = confidence
        self.outcome_code = outcome_code
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.fallback_used = fallback_used
        self.generation_time_ms = generation_time_ms


def _conversation_summary(
    conversation: ChatConversation,
    message_count: int,
) -> ConversationSummaryDTO:
    return ConversationSummaryDTO(
        conversation_id=conversation.conversation_id,
        job_id=conversation.job_id,
        title=conversation.title,
        created_by=conversation.created_by,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=message_count,
    )


def _conversation_detail(
    conversation: ChatConversation,
    messages: Sequence[ChatMessage],
) -> ConversationDetailDTO:
    return ConversationDetailDTO(
        **_conversation_summary(conversation, len(messages)).model_dump(),
        messages=[_message_dto(message) for message in messages],
    )


def _message_dto(message: ChatMessage) -> ChatMessageDTO:
    evidence = message.evidence
    evidence_dto = (
        ChatTurnEvidenceDTO(
            citations=evidence.citations_json,
            data_evidence=evidence.data_evidence_json,
            tool_calls=evidence.tool_calls_json,
            limitations=evidence.limitations_json,
            llm_provider=evidence.llm_provider,
            llm_model=evidence.llm_model,
            fallback_used=evidence.fallback_used,
            generation_time_ms=evidence.generation_time_ms,
            prompt_template_id=evidence.prompt_template_id,
            prompt_template_sha256=evidence.prompt_template_sha256,
        )
        if evidence
        else None
    )
    return ChatMessageDTO(
        message_id=message.message_id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        query_type=QueryType(message.query_type),
        image_id=message.image_id,
        run_ids=message.run_ids_json,
        material_context=(
            MaterialContext.model_validate(message.material_context_json)
            if message.material_context_json
            else None
        ),
        confidence=message.confidence,
        outcome_code=message.outcome_code,
        evidence=evidence_dto,
        created_at=message.created_at,
    )


def _bounded_history(
    messages: Sequence[ChatMessage],
    *,
    turns: int,
    max_chars: int,
) -> list[Mapping[str, str]]:
    candidates = [
        message
        for message in messages
        if message.role in {"user", "assistant"} and message.content.strip()
    ][-(turns * 2) :]
    selected: list[Mapping[str, str]] = []
    used = 0
    for message in reversed(candidates):
        remaining = max_chars - used
        if remaining <= 0:
            break
        content = message.content[-remaining:]
        selected.append({"role": message.role, "content": content})
        used += len(content)
    selected.reverse()
    return selected


def _cite_deterministic_data(answer: str, evidence_count: int) -> str:
    if not answer or evidence_count <= 0:
        return answer
    markers = "".join(f"[D{index}]" for index in range(1, evidence_count + 1))
    return "".join(
        (
            f"{sentence.rstrip()} {markers}"
            if _NUMBER.search(sentence) and "[D" not in sentence
            else sentence
        )
        for sentence in _NUMERIC_SENTENCE.findall(answer)
    ).strip()


def _analysis_guardrail_evidence(
    evidence: Sequence[ToolEvidence],
) -> ToolEvidence:
    rows = [row for item in evidence for row in item.rows]
    quality_statuses = {
        str(row.get("quality_status")) for row in rows if row.get("quality_status") is not None
    }
    physical_scale_available = any(
        row.get("number_density_um2") is not None
        or row.get("mean_equivalent_diameter_nm") is not None
        for row in rows
    )
    quality_interpretation = (
        "自动质量门控通过，但不能替代人工实例边界抽查"
        if quality_statuses == {"PASS"}
        else "自动质量门控存在告警或状态不完整，应先人工复核实例边界与告警区域"
    )
    scale_interpretation = (
        "当前证据包含物理尺度指标，不得声称缺少比例尺"
        if physical_scale_available
        else "当前证据只支持像素尺度，不得推断物理尺寸"
    )
    source_run_ids = sorted({run_id for item in evidence for run_id in item.source_run_ids})
    warnings = list(
        dict.fromkeys(warning for item in evidence for warning in item.quality_warnings)
    )
    return ToolEvidence(
        tool_name="interpret_analysis_guardrails",
        validated_arguments={"intent": "qualitative_interpretation"},
        aggregates={
            "quality_interpretation": quality_interpretation,
            "scale_interpretation": scale_interpretation,
            "scope_limitation": "当前图像视野不能单独代表完整样品的空间异质性",
            "allowed_next_steps": [
                "人工抽查小颗粒、粘连区和边界实例",
                "增加重复视野以评估空间变异",
                "增加对照运行或人工真值以评估模型与参数敏感性",
            ],
        },
        source_run_ids=source_run_ids,
        quality_warnings=warnings,
    )


def _validate_qualitative_analysis_synthesis(
    *,
    answer: str,
    physical_scale_available: bool,
) -> None:
    if _NUMBER.search(_EVIDENCE_REFERENCE.sub("", answer)):
        raise CitationValidationError(
            "analysis synthesis must leave numeric claims to deterministic tools"
        )
    if not _DATA_REFERENCE.search(answer):
        raise CitationValidationError(
            "analysis synthesis requires at least one data evidence reference"
        )
    scale_claims = _NEGATED_MISSING_SCALE_CLAIM.sub("", answer)
    if physical_scale_available and _MISSING_SCALE_CLAIM.search(scale_claims):
        raise CitationValidationError("analysis synthesis contradicted the frozen physical scale")


def _guardrail_has_physical_scale(evidence: Sequence[ToolEvidence]) -> bool:
    return any(
        item.tool_name == "interpret_analysis_guardrails"
        and str(item.aggregates.get("scale_interpretation", "")).startswith(
            "当前证据包含物理尺度指标"
        )
        for item in evidence
    )


def _deterministic_analysis_limitations(
    evidence: Sequence[ToolEvidence],
) -> tuple[str, ...]:
    for index, item in enumerate(evidence, start=1):
        if item.tool_name != "interpret_analysis_guardrails":
            continue
        marker = f"[D{index}]"
        return tuple(
            f"{text} {marker}"
            for text in (
                item.aggregates.get("quality_interpretation"),
                item.aggregates.get("scope_limitation"),
            )
            if isinstance(text, str) and text
        )
    return ()


def _merge_research_evidence(
    knowledge: KnowledgeEvidence,
    research: ResearchEvidence,
) -> KnowledgeEvidence:
    chunks = [
        *(context.chunk for context in knowledge.contexts),
        *research.chunks,
    ][:12]
    contexts: list[CitationContext] = []
    citations: list[Citation] = []
    for index, chunk in enumerate(chunks, start=1):
        citation_id = f"C{index}"
        contexts.append(CitationContext(citation_id=citation_id, chunk=chunk))
        citations.append(_citation_from_chunk(citation_id, chunk))
    return KnowledgeEvidence(
        contexts=tuple(contexts),
        citations=tuple(citations),
        limitations=tuple(
            dict.fromkeys(
                [
                    *knowledge.limitations,
                    *research.limitations,
                    "联网检索结果是本轮发现的摘要或题录，不等同于完整、系统的文献综述",
                ]
            )
        ),
        outcome_code="OK" if contexts else "INSUFFICIENT_EVIDENCE",
        blocked_untrusted_instruction=knowledge.blocked_untrusted_instruction,
    )


def _citation_from_chunk(citation_id: str, chunk: RetrievedChunk) -> Citation:
    compact = " ".join(chunk.text.split())
    excerpt = compact if len(compact) <= 160 else compact[:159].rstrip() + "…"
    return Citation(
        citation_id=citation_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        page=chunk.page_start,
        chunk_id=chunk.chunk_id,
        excerpt=excerpt,
        retrieval_score=chunk.retrieval_score,
        source_type=chunk.source_type,
        citation_text=chunk.citation_text,
        url=chunk.source_url,
    )


def _outcome(
    query_type: QueryType,
    data: DataQueryResult,
    knowledge: KnowledgeEvidence,
) -> str:
    if query_type is QueryType.GENERAL_CHAT:
        return "OK"
    if query_type is QueryType.ANALYSIS_DATA:
        return data.outcome_code
    if query_type is QueryType.MATERIAL_KNOWLEDGE:
        return knowledge.outcome_code
    return (
        "OK"
        if data.outcome_code == "OK" and knowledge.outcome_code == "OK"
        else "INSUFFICIENT_EVIDENCE"
    )


def _title_from_message(content: str) -> str:
    compact = " ".join(content.split())
    return compact[:40] + ("…" if len(compact) > 40 else "")


def _data_tool_question(
    question: str,
    *,
    previous_user_question: str | None,
) -> str:
    normalized = question.casefold().strip()
    if not any(marker in normalized for marker in ("差异", "为什么", "那", "这个")):
        return question
    return f"{previous_user_question}；{question}" if previous_user_question else question


def _elapsed_ms(started: float) -> int:
    return max(0, round((monotonic() - started) * 1_000))


def _policy_refusal(started: float) -> _TurnAnswer:
    return _TurnAnswer(
        content=("我不能忽略证据或编造科学结论。请提供可核验的当前任务数据或知识库来源。"),
        data=DataQueryResult(answer=""),
        tool_calls=(),
        citations=(),
        limitations=("拒绝绕过数据与引用约束；未调用模型或数据/检索工具",),
        confidence="low",
        outcome_code="INSUFFICIENT_EVIDENCE",
        llm_provider="policy",
        llm_model=None,
        fallback_used=False,
        generation_time_ms=_elapsed_ms(started),
    )


def _tenant_id(principal: PrincipalContext) -> str:
    if principal.tenant_id is None:
        raise ValueError("principal must carry a tenant ID")
    return principal.tenant_id


def _principal_id(principal: PrincipalContext) -> str:
    if principal.principal_id is None:
        raise ValueError("principal must carry a principal ID")
    return principal.principal_id
