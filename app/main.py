"""
FastAPI main application for Voice RAG.

Endpoints:
- POST /voice-query      → Full voice pipeline (audio → STT → RAG → answer)
- POST /text-query       → Text-only RAG query
- GET  /health           → Health check
- GET  /latency-report   → Latency analytics (P50/P70/P100)
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.models import (
    RAGQueryRequest,
    RAGResponse,
    VoiceQueryResponse,
    LatencyMetrics,
    RetrievedPassage,
)
from app.stt import SarvamSTT, MockSTT
from app.rag_pipeline import RAGPipeline
from app.llm_handler import LLMHandler
from app.guardrails import Guardrails


# ============================================================
# GLOBAL STATE
# ============================================================

rag_pipeline: Optional[RAGPipeline] = None
stt_client = None
llm_handler: Optional[LLMHandler] = None
guardrails: Optional[Guardrails] = None
latency_history: list = []  # Store latency measurements


# ============================================================
# LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    global rag_pipeline, stt_client, llm_handler, guardrails
    
    print("🚀 Initializing Voice RAG pipeline...")
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    index_path = os.getenv("INDEX_PATH", "data/rag_index")
    local_json = os.getenv("LOCAL_DATASET_PATH", "data/hi_subset_5000.json")
    max_samples = int(os.getenv("MAX_SAMPLES", "5000"))
    
    if os.path.exists(f"{index_path}.faiss"):
        print("Loading existing FAISS index...")
        rag_pipeline._load_index(index_path)
    else:
        print("Building new index from dataset...")
        rag_pipeline.load_and_index_dataset(
            language="hi",
            strategies=["fixed_size", "semantic_sentence", "passage_level", "metadata_aware", "sliding_window"],
            save_path=index_path,
            local_json_path=local_json if os.path.exists(local_json) else None,
            max_samples=max_samples,
        )
    
    # Initialize STT
    sarvam_key = os.getenv("SARVAM_API_KEY")
    if sarvam_key and sarvam_key != "your_sarvam_api_key_here":
        stt_client = SarvamSTT(api_key=sarvam_key)
        print("✅ Sarvam STT initialized")
    else:
        stt_client = MockSTT()
        print("⚠️ Using Mock STT (set SARVAM_API_KEY for real STT)")
    
    # Initialize LLM
    llm_handler = LLMHandler()
    if os.getenv("GROQ_API_KEY"):
        print("✅ Groq LLM initialized")
    else:
        print("⚠️ Using extractive fallback (set GROQ_API_KEY for LLM generation)")
    
    # Initialize guardrails
    guardrails = Guardrails()
    print("✅ Guardrails initialized")
    
    print("🎉 Voice RAG pipeline ready!")
    
    yield
    
    # Cleanup
    if hasattr(stt_client, "close"):
        await stt_client.close()
    if llm_handler:
        await llm_handler.close()
    print("Shutdown complete.")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Voice RAG - HH Goa 2026",
    description="Voice-enabled Retrieval-Augmented Generation system for multilingual QA",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_ready": rag_pipeline.is_ready if rag_pipeline else False,
        "stt_type": type(stt_client).__name__ if stt_client else "None",
        "llm_available": llm_handler.client is not None if llm_handler else False,
    }


@app.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(
    audio_file: UploadFile = File(...),
    language_code: str = Form(default="hi-IN"),
    top_k: int = Form(default=5),
):
    """
    Full voice pipeline:
    Audio → STT → Guardrails → RAG Retrieval → LLM Generation → Guardrails → Response
    """
    total_start = time.perf_counter()
    
    # 1. Speech-to-Text
    audio_bytes = await audio_file.read()
    stt_response = await stt_client.transcribe(
        audio_bytes=audio_bytes,
        language_code=language_code,
        filename=audio_file.filename or "audio.wav",
        content_type=audio_file.content_type or "audio/wav",
    )
    
    # 2. Input Guardrails
    input_check = guardrails.check_input(stt_response.transcript)
    if not input_check.is_safe:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "blocked_by_guardrails",
                "reason": input_check.blocked_reason,
                "flags": input_check.flags,
            },
        )
    
    # 3. RAG Retrieval
    passages, retrieval_latency = rag_pipeline.retrieve(
        query=stt_response.transcript,
        top_k=top_k,
    )
    
    # 4. Retrieval Guardrails
    retrieval_check = guardrails.check_retrieval(stt_response.transcript, passages)
    
    # 5. LLM Generation
    if retrieval_check.is_on_topic:
        try:
            import langdetect
            detected_lang = langdetect.detect(stt_response.transcript)
        except:
            detected_lang = language_code.split("-")[0]
            
        llm_result = await llm_handler.generate(
            query=stt_response.transcript,
            passages=passages,
            language=detected_lang,
        )
    else:
        llm_result = {
            "answer": retrieval_check.blocked_reason or "No relevant information found.",
            "confidence": 0.0,
            "sources_used": [],
            "latency_ms": 0.0,
        }
    
    # 6. Output Guardrails
    output_check = guardrails.check_output(llm_result["answer"], passages)
    
    total_latency = (time.perf_counter() - total_start) * 1000
    
    # Track latency
    latency_breakdown = {
        "stt_ms": stt_response.latency_ms,
        "retrieval_ms": retrieval_latency,
        "generation_ms": llm_result["latency_ms"],
        "total_ms": total_latency,
    }
    latency_history.append(latency_breakdown)
    
    # Build response
    rag_response = RAGResponse(
        answer=llm_result["answer"],
        sources=passages,
        query_original=stt_response.transcript,
        confidence=llm_result["confidence"],
        is_grounded=output_check.is_grounded,
        guardrail_flags=list(set(
            input_check.flags + retrieval_check.flags + output_check.flags
        )),
        latency_breakdown=latency_breakdown,
    )
    
    return VoiceQueryResponse(
        stt=stt_response,
        rag=rag_response,
        total_latency_ms=total_latency,
    )


@app.post("/text-query", response_model=RAGResponse)
async def text_query(request: RAGQueryRequest):
    """
    Text-only RAG query (no STT).
    Useful for testing and text-based interactions.
    """
    total_start = time.perf_counter()
    
    # 1. Input Guardrails
    input_check = guardrails.check_input(request.query)
    if not input_check.is_safe:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "blocked_by_guardrails",
                "reason": input_check.blocked_reason,
                "flags": input_check.flags,
            },
        )
    
    # 2. RAG Retrieval
    passages, retrieval_latency = rag_pipeline.retrieve(
        query=request.query,
        top_k=request.top_k,
    )
    
    # 3. Retrieval Guardrails
    retrieval_check = guardrails.check_retrieval(request.query, passages)
    
    # 4. LLM Generation
    if retrieval_check.is_on_topic:
        try:
            import langdetect
            detected_lang = langdetect.detect(request.query)
        except:
            detected_lang = request.language.value
            
        llm_result = await llm_handler.generate(
            query=request.query,
            passages=passages,
            language=detected_lang,
        )
    else:
        llm_result = {
            "answer": retrieval_check.blocked_reason or "No relevant information found.",
            "confidence": 0.0,
            "sources_used": [],
            "latency_ms": 0.0,
        }
    
    # 5. Output Guardrails
    output_check = guardrails.check_output(llm_result["answer"], passages)
    
    total_latency = (time.perf_counter() - total_start) * 1000
    
    # Track latency
    latency_breakdown = {
        "stt_ms": 0.0,
        "retrieval_ms": retrieval_latency,
        "generation_ms": llm_result["latency_ms"],
        "total_ms": total_latency,
    }
    latency_history.append(latency_breakdown)
    
    return RAGResponse(
        answer=llm_result["answer"],
        sources=passages,
        query_original=request.query,
        confidence=llm_result["confidence"],
        is_grounded=output_check.is_grounded,
        guardrail_flags=list(set(
            input_check.flags + retrieval_check.flags + output_check.flags
        )),
        latency_breakdown=latency_breakdown,
    )


@app.get("/latency-report", response_model=LatencyMetrics)
async def latency_report():
    """
    Get P50/P70/P100 latency analytics.
    """
    if not latency_history:
        raise HTTPException(status_code=404, detail="No latency data available yet. Run some queries first.")
    
    import numpy as np
    
    totals = [h["total_ms"] for h in latency_history]
    
    # Per-stage breakdown
    stages = {}
    for stage in ["stt_ms", "retrieval_ms", "generation_ms"]:
        values = [h.get(stage, 0) for h in latency_history]
        stages[stage] = {
            "p50": float(np.percentile(values, 50)),
            "p70": float(np.percentile(values, 70)),
            "p100": float(np.percentile(values, 100)),
            "mean": float(np.mean(values)),
        }
    
    return LatencyMetrics(
        p50_ms=float(np.percentile(totals, 50)),
        p70_ms=float(np.percentile(totals, 70)),
        p100_ms=float(np.percentile(totals, 100)),
        mean_ms=float(np.mean(totals)),
        num_queries=len(latency_history),
        breakdown=stages,
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the premium frontend UI."""
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
