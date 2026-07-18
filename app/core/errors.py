"""Domain exceptions and the frozen error-code to HTTP mapping."""

from collections.abc import Mapping
from typing import Any


class NanoLoopError(Exception):
    code = "INTERNAL_ERROR"
    status_code = 500
    retryable = False
    default_message = "服务器内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.details = dict(details or {})
        if retryable is not None:
            self.retryable = retryable


class InvalidImageError(NanoLoopError):
    code = "INVALID_IMAGE"
    status_code = 400
    default_message = "图像文件无效或无法读取"


class MissingScaleError(NanoLoopError):
    code = "MISSING_SCALE"
    status_code = 422
    default_message = "当前操作需要有效的物理比例尺"


class InvalidBoxError(NanoLoopError):
    code = "INVALID_BOX"
    status_code = 422
    default_message = "矩形框无效"


class BoxRevisionConflictError(NanoLoopError):
    code = "BOX_REVISION_CONFLICT"
    status_code = 409
    default_message = "矩形框版本已更新，请重新读取"


class ModelNotFoundError(NanoLoopError):
    code = "MODEL_NOT_FOUND"
    status_code = 404
    default_message = "找不到指定模型"


class ModelNotReadyError(NanoLoopError):
    code = "MODEL_NOT_READY"
    status_code = 503
    default_message = "模型当前不可用"


class InferenceExecutionError(NanoLoopError):
    code = "INFERENCE_FAILED"
    status_code = 500
    retryable = True
    default_message = "模型推理失败"


class InputArtifactMismatchError(NanoLoopError):
    code = "INPUT_ARTIFACT_MISMATCH"
    status_code = 409
    default_message = "分析输入文件与运行快照不一致"


class ExecutionBuildMismatchError(NanoLoopError):
    code = "EXECUTION_BUILD_MISMATCH"
    status_code = 409
    default_message = "执行节点的软件构建与运行快照不一致"


class RagIndexNotReadyError(NanoLoopError):
    code = "RAG_INDEX_NOT_READY"
    status_code = 503
    default_message = "知识索引尚未就绪"


class InsufficientEvidenceError(NanoLoopError):
    code = "INSUFFICIENT_EVIDENCE"
    status_code = 200
    default_message = "知识库证据不足"


class JobStateConflictError(NanoLoopError):
    code = "JOB_STATE_CONFLICT"
    status_code = 409
    default_message = "当前任务状态不允许该操作"


class ExportNotReadyError(NanoLoopError):
    code = "EXPORT_NOT_READY"
    status_code = 409
    default_message = "任务结果尚未达到可导出状态"


class ResourceNotFoundError(NanoLoopError):
    code = "RESOURCE_NOT_FOUND"
    status_code = 404
    default_message = "找不到指定资源"


class StorageError(NanoLoopError):
    code = "STORAGE_ERROR"
    status_code = 500
    retryable = True
    default_message = "文件存储操作失败"


class ApiNotImplementedError(NanoLoopError):
    """Raised when a frozen API surface exists before its application service."""

    code = "NOT_IMPLEMENTED"
    status_code = 501
    default_message = "该能力的应用服务尚未接入"


class ServiceUnavailableError(NanoLoopError):
    code = "SERVICE_UNAVAILABLE"
    status_code = 503
    retryable = True
    default_message = "依赖服务当前不可用"


class PayloadTooLargeError(NanoLoopError):
    code = "PAYLOAD_TOO_LARGE"
    status_code = 413
    default_message = "上传内容超过允许大小"


class UnsupportedMediaTypeError(NanoLoopError):
    code = "UNSUPPORTED_MEDIA_TYPE"
    status_code = 415
    default_message = "不支持该文件类型"


class InvalidMultipartError(NanoLoopError):
    code = "INVALID_MULTIPART"
    status_code = 400
    default_message = "multipart 上传内容不符合接口约束"


class InvalidKnowledgeDocumentError(NanoLoopError):
    code = "INVALID_KNOWLEDGE_DOCUMENT"
    status_code = 400
    default_message = "知识文档无效或无法提取"


class KnowledgeDocumentConflictError(NanoLoopError):
    code = "KNOWLEDGE_DOCUMENT_CONFLICT"
    status_code = 409
    default_message = "相同文档内容已使用不同元数据入库"


class KnowledgeDocumentStateConflictError(NanoLoopError):
    code = "KNOWLEDGE_DOCUMENT_STATE_CONFLICT"
    status_code = 409
    default_message = "知识文档当前状态不允许该操作"
