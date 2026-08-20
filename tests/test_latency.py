"""
Latency testing module.

Measures P50/P70/P100 latency across the RAG pipeline
using a batch of test queries.
"""

import time
import asyncio
import numpy as np
from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag_pipeline import RAGPipeline
from app.llm_handler import LLMHandler
from app.guardrails import Guardrails
from app.models import LatencyMetrics


# Test queries in Hindi (covering diverse topics)
TEST_QUERIES = [
    "मेनहट्टन प्रोजेक्ट क्या था?",
    "भारत की राजधानी क्या है?",
    "सूर्य कितना बड़ा है?",
    "पानी का रासायनिक सूत्र क्या है?",
    "महात्मा गांधी कौन थे?",
    "चंद्रमा पृथ्वी से कितनी दूर है?",
    "ताजमहल कहाँ स्थित है?",
    "कंप्यूटर का आविष्कार किसने किया?",
    "विश्व का सबसे बड़ा महासागर कौन सा है?",
    "भारत में कितने राज्य हैं?",
    "इंटरनेट कैसे काम करता है?",
    "हिमालय कहाँ स्थित है?",
    "पृथ्वी की आयु कितनी है?",
    "सौर मंडल में कितने ग्रह हैं?",
    "मानव शरीर में कितनी हड्डियाँ होती हैं?",
    "DNA क्या है?",
    "विश्व युद्ध कब हुआ था?",
    "ऑक्सीजन कैसे बनती है?",
    "चाँद पर कौन गया था?",
    "भारत का संविधान कब लागू हुआ?",
]


async def run_latency_test(
    num_queries: int = 20,
    top_k: int = 5,
    dataset_split: str = "train[:5000]",
) -> LatencyMetrics:
    """
    Run latency test across multiple queries.
    Returns P50, P70, P100 latency metrics.
    """
    print("=" * 60)
    print("LATENCY TEST")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    
    index_path = "data/rag_index"
    if os.path.exists(f"{index_path}.faiss"):
        pipeline._load_index(index_path)
    else:
        pipeline.load_and_index_dataset(
            language="hi",
            split=dataset_split,
            save_path=index_path,
            local_json_path="data/hi_subset_5000.json",
            max_samples=100,
            strategies=["passage_level"],
        )
    
    # Initialize other components
    llm = LLMHandler()
    guard = Guardrails()
    
    # Run queries
    queries = TEST_QUERIES[:num_queries]
    print(f"\n2. Running {len(queries)} test queries...\n")
    
    latency_records = {
        "retrieval_ms": [],
        "guardrail_ms": [],
        "generation_ms": [],
        "total_ms": [],
    }
    
    for i, query in enumerate(queries):
        total_start = time.perf_counter()
        
        # Guardrails (input)
        guard_start = time.perf_counter()
        input_check = guard.check_input(query)
        guard_latency = (time.perf_counter() - guard_start) * 1000
        
        # Retrieval
        passages, retrieval_latency = pipeline.retrieve(query, top_k=top_k)
        
        # Retrieval guardrails
        guard_start2 = time.perf_counter()
        retrieval_check = guard.check_retrieval(query, passages)
        guard_latency += (time.perf_counter() - guard_start2) * 1000
        
        # LLM generation (extractive fallback if no API key)
        llm_result = await llm.generate(query, passages, language="hi")
        
        # Output guardrails
        guard_start3 = time.perf_counter()
        output_check = guard.check_output(llm_result["answer"], passages)
        guard_latency += (time.perf_counter() - guard_start3) * 1000
        
        total_latency = (time.perf_counter() - total_start) * 1000
        
        latency_records["retrieval_ms"].append(retrieval_latency)
        latency_records["guardrail_ms"].append(guard_latency)
        latency_records["generation_ms"].append(llm_result["latency_ms"])
        latency_records["total_ms"].append(total_latency)
        
        status = "✅" if total_latency < 200 else "⚠️"
        print(f"  {status} Query {i+1:2d}: {total_latency:7.1f}ms total | "
              f"retrieval: {retrieval_latency:5.1f}ms | "
              f"gen: {llm_result['latency_ms']:5.1f}ms | "
              f"guard: {guard_latency:5.1f}ms")
    
    # Calculate percentiles
    totals = latency_records["total_ms"]
    
    breakdown = {}
    for stage in ["retrieval_ms", "guardrail_ms", "generation_ms"]:
        values = latency_records[stage]
        breakdown[stage] = {
            "p50": float(np.percentile(values, 50)),
            "p70": float(np.percentile(values, 70)),
            "p100": float(np.percentile(values, 100)),
            "mean": float(np.mean(values)),
        }
    
    metrics = LatencyMetrics(
        p50_ms=float(np.percentile(totals, 50)),
        p70_ms=float(np.percentile(totals, 70)),
        p100_ms=float(np.percentile(totals, 100)),
        mean_ms=float(np.mean(totals)),
        num_queries=len(queries),
        breakdown=breakdown,
    )
    
    # Print report
    print("\n" + "=" * 60)
    print("LATENCY REPORT")
    print("=" * 60)
    print(f"  Queries tested: {metrics.num_queries}")
    print(f"  P50  (median):  {metrics.p50_ms:.1f} ms")
    print(f"  P70:            {metrics.p70_ms:.1f} ms")
    print(f"  P100 (max):     {metrics.p100_ms:.1f} ms")
    print(f"  Mean:           {metrics.mean_ms:.1f} ms")
    print(f"\n  Target: < 200ms")
    print(f"  Status: {'✅ PASS' if metrics.p100_ms < 200 else '⚠️ Needs optimization'}")
    
    print("\n  Per-stage breakdown:")
    for stage, stats in breakdown.items():
        print(f"    {stage:20s}: P50={stats['p50']:6.1f}ms | P70={stats['p70']:6.1f}ms | P100={stats['p100']:6.1f}ms")
    print("=" * 60)
    
    await llm.close()
    
    return metrics


if __name__ == "__main__":
    metrics = asyncio.run(run_latency_test(num_queries=20))
