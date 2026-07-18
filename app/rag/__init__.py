"""Evidence-grounded material knowledge retrieval and answer generation."""

from app.rag.application import (
    DuplicateKnowledgeDocumentError,
    KnowledgeApplicationError,
    KnowledgeApplicationService,
    KnowledgeDocumentNotFoundError,
    KnowledgeDocumentStateError,
    KnowledgeIndexUnavailableError,
    KnowledgeSourcePathError,
)
from app.rag.embeddings import SentenceTransformerEmbeddingProvider
from app.rag.providers import ExtractiveAnswerProvider, OpenAICompatibleProvider
from app.rag.retrieval import RetrievalReport, RetrievalService
from app.rag.service import KnowledgeAnswer, KnowledgeService
from app.rag.vector_index import DatabaseVectorIndexPublisher
from app.rag.vector_store import PersistentFaissVectorStore

__all__ = [
    "DatabaseVectorIndexPublisher",
    "DuplicateKnowledgeDocumentError",
    "ExtractiveAnswerProvider",
    "KnowledgeAnswer",
    "KnowledgeApplicationError",
    "KnowledgeApplicationService",
    "KnowledgeDocumentNotFoundError",
    "KnowledgeDocumentStateError",
    "KnowledgeIndexUnavailableError",
    "KnowledgeService",
    "KnowledgeSourcePathError",
    "OpenAICompatibleProvider",
    "PersistentFaissVectorStore",
    "RetrievalReport",
    "RetrievalService",
    "SentenceTransformerEmbeddingProvider",
]
