# 🎙️ Voice RAG - HH Goa 2026

A voice-enabled Retrieval-Augmented Generation (RAG) system for multilingual QA, built on the [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset.

## 🏗️ Architecture

```
Voice Input → Sarvam STT → Guardrails → RAG Retrieval (FAISS) → LLM Generation (Groq) → Guardrails → Response
```

## 🚀 Features

### Speech-to-Text
- **Sarvam AI** integration for Hindi and other Indic languages
- Retry logic with exponential backoff
- Mock STT for development/testing

### Chunking Strategies (5 approaches)
1. **Fixed-size chunking** — Consistent chunk sizes with configurable overlap
2. **Semantic sentence chunking** — Splits on sentence boundaries (Hindi & English)
3. **Passage-level chunking** — Uses native MSMARCO passage boundaries
4. **Metadata-aware chunking** — Preserves query-passage relationships
5. **Sliding window chunking** — Dynamic sizing based on content density

### Vector Retrieval
- FAISS indexing with cosine similarity
- Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- Persistent index storage

### LLM Generation
- **Groq** for ultra-fast inference (Llama 3.1 8B Instant)
- Structured JSON output
- Extractive fallback when LLM unavailable

### Guardrails
- Off-topic query detection
- Unsafe/inappropriate input filtering
- Prompt injection detection
- Hallucination checks (grounding verification)
- Confidence-based answer gating

### Model Harness
- Structured orchestration via FastAPI
- Tool calls and retries
- Structured I/O with Pydantic
- Error recovery (extractive fallback)

### Latency Analytics
- P50 / P70 / P100 latency reporting
- Per-stage breakdown (STT, retrieval, generation, guardrails)
- Batch latency testing tool

## 📁 Project Structure

```
voice-rag-hh-goa/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app with full pipeline
│   ├── stt.py            # Sarvam STT integration
│   ├── rag_pipeline.py   # Chunking, indexing, retrieval
│   ├── llm_handler.py    # Groq LLM integration
│   ├── guardrails.py     # Safety and relevance checks
│   └── models.py         # Pydantic schemas
├── data/
│   └── sample_hindi.json
├── tests/
│   └── test_latency.py   # Latency measurement tool
├── explore_dataset.py    # Dataset exploration
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🛠️ Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API keys
export SARVAM_API_KEY='your_key'
export GROQ_API_KEY='your_key'
```

## 🏃 Running

### 1. Explore the dataset
```bash
python explore_dataset.py
```

### 2. Start the API server
```bash
python -m app.main
# or
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run latency tests
```bash
python -m tests.test_latency
```

### 4. Test with curl
```bash
# Text query
curl -X POST http://localhost:8000/text-query \
  -H "Content-Type: application/json" \
  -d '{"query": "मेनहट्टन प्रोजेक्ट क्या था?", "language": "hi", "top_k": 5}'

# Voice query
curl -X POST http://localhost:8000/voice-query \
  -F "audio_file=@sample.wav" \
  -F "language_code=hi-IN"

# Latency report
curl http://localhost:8000/latency-report
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Info page |
| `/health` | GET | Health check |
| `/voice-query` | POST | Full voice pipeline |
| `/text-query` | POST | Text-only RAG query |
| `/latency-report` | GET | P50/P70/P100 latency |
| `/docs` | GET | Swagger API docs |

## 🏆 HH Goa 2026 - Task 2

Built for the HH Goa 2026 shortlisting task. #RAGInGoa

### Requirements Met
- ✅ Speech-to-text (Sarvam AI)
- ✅ Vast chunking strategy (5 approaches)
- ✅ Latency target (<200ms)
- ✅ Latency analytics (P50/P70/P100)
- ✅ Model harness (structured orchestration)
- ✅ Guardrails (safety, relevance, grounding)
# hhgoa_tast2
