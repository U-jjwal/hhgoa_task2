"""Pydantic schemas for structured input/output handling."""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class QueryLanguage(str, Enum):
    HINDI = "hi"
    ENGLISH = "en"
    TAMIL = "ta"
    TELUGU = "te"
    BENGALI = "bn"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    PUNJABI = "pa"
    ODIA = "or"


class STTRequest(BaseModel):
    """Request model for Speech-to-Text."""
    language_code: str = Field(default="hi-IN", description="Language code for STT")


class STTResponse(BaseModel):
    """Response model for Speech-to-Text."""
    transcript: str
    language_detected: str
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float


class RetrievedPassage(BaseModel):
    """A single retrieved passage with metadata."""
    text: str
    score: float
    chunk_strategy: str  # Which chunking strategy found this
    source_index: int


class RAGQueryRequest(BaseModel):
    """Request model for text-based RAG query."""
    query: str
    language: QueryLanguage = QueryLanguage.HINDI
    top_k: int = Field(default=5, ge=1, le=20)


class RAGResponse(BaseModel):
    """Structured response from the RAG pipeline."""
    answer: str
    sources: List[RetrievedPassage]
    query_original: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_grounded: bool  # Whether answer is grounded in retrieved context
    guardrail_flags: List[str] = Field(default_factory=list)
    latency_breakdown: dict  # stt_ms, retrieval_ms, generation_ms, total_ms


class VoiceQueryResponse(BaseModel):
    """Full voice query response including STT + RAG."""
    stt: STTResponse
    rag: RAGResponse
    total_latency_ms: float


class GuardrailResult(BaseModel):
    """Result of guardrail checks."""
    is_safe: bool
    is_on_topic: bool
    is_grounded: bool
    flags: List[str] = Field(default_factory=list)
    blocked_reason: Optional[str] = None


class LatencyMetrics(BaseModel):
    """Latency analytics for the pipeline."""
    p50_ms: float
    p70_ms: float
    p100_ms: float
    mean_ms: float
    num_queries: int
    breakdown: dict  # Per-stage latency stats
