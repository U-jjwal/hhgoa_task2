"""
RAG Pipeline with multiple chunking strategies.

Chunking strategies implemented:
1. Fixed-size chunking with overlap
2. Semantic sentence-based chunking
3. Passage-level chunking (dataset native)
4. Metadata-aware chunking (preserving query-passage relationships)
5. Sliding window chunking with dynamic sizing
"""

import os
import time
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from app.models import RetrievedPassage


# ============================================================
# CHUNKING STRATEGIES
# ============================================================

@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    strategy: str  # Which strategy produced this chunk
    source_idx: int  # Index in the original dataset
    metadata: dict = field(default_factory=dict)


class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def chunk(self, passages: List[str], metadata: Optional[List[dict]] = None) -> List[Chunk]:
        raise NotImplementedError


class FixedSizeChunker(ChunkingStrategy):
    """
    Strategy 1: Fixed-size chunking with configurable overlap.
    Good for: Consistent chunk sizes for embedding models.
    """
    
    def __init__(self, chunk_size: int = 256, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, passages: List[str], metadata: Optional[List[dict]] = None) -> List[Chunk]:
        chunks = []
        for idx, passage in enumerate(passages):
            words = passage.split()
            if len(words) <= self.chunk_size:
                chunks.append(Chunk(
                    text=passage,
                    strategy="fixed_size",
                    source_idx=idx,
                    metadata=metadata[idx] if metadata else {},
                ))
            else:
                # Split into overlapping windows
                start = 0
                while start < len(words):
                    end = min(start + self.chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    chunks.append(Chunk(
                        text=chunk_text,
                        strategy="fixed_size",
                        source_idx=idx,
                        metadata={
                            **(metadata[idx] if metadata else {}),
                            "window_start": start,
                            "window_end": end,
                        },
                    ))
                    if end >= len(words):
                        break
                    start += self.chunk_size - self.overlap
        return chunks


class SemanticSentenceChunker(ChunkingStrategy):
    """
    Strategy 2: Semantic sentence-based chunking.
    Splits on sentence boundaries, groups sentences into chunks
    that maintain semantic coherence.
    """
    
    def __init__(self, max_sentences: int = 5, min_chunk_length: int = 50):
        self.max_sentences = max_sentences
        self.min_chunk_length = min_chunk_length
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (handles Hindi and English)."""
        # Hindi sentence endings: ।, |, ?, !
        # English sentence endings: ., ?, !
        import re
        sentences = re.split(r'(?<=[।|.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, passages: List[str], metadata: Optional[List[dict]] = None) -> List[Chunk]:
        chunks = []
        for idx, passage in enumerate(passages):
            sentences = self._split_sentences(passage)
            
            if len(sentences) <= self.max_sentences:
                chunks.append(Chunk(
                    text=passage,
                    strategy="semantic_sentence",
                    source_idx=idx,
                    metadata=metadata[idx] if metadata else {},
                ))
            else:
                # Group sentences into chunks
                for i in range(0, len(sentences), self.max_sentences - 1):
                    group = sentences[i:i + self.max_sentences]
                    chunk_text = " ".join(group)
                    if len(chunk_text) >= self.min_chunk_length:
                        chunks.append(Chunk(
                            text=chunk_text,
                            strategy="semantic_sentence",
                            source_idx=idx,
                            metadata={
                                **(metadata[idx] if metadata else {}),
                                "sentence_range": f"{i}-{i + len(group)}",
                            },
                        ))
        return chunks


class PassageLevelChunker(ChunkingStrategy):
    """
    Strategy 3: Passage-level chunking (dataset native).
    Uses the natural passage boundaries from the MSMARCO dataset.
    """
    
    def chunk(self, passages: List[str], metadata: Optional[List[dict]] = None) -> List[Chunk]:
        chunks = []
        for idx, passage in enumerate(passages):
            if passage.strip():
                chunks.append(Chunk(
                    text=passage.strip(),
                    strategy="passage_level",
                    source_idx=idx,
                    metadata=metadata[idx] if metadata else {},
                ))
        return chunks


class MetadataAwareChunker(ChunkingStrategy):
    """
    Strategy 4: Metadata-aware chunking.
    Preserves query-passage relationships and selection status.
    Prepends query context to improve retrieval accuracy.
    """
    
    def chunk(self, passages: List[str], metadata: Optional[List[dict]] = None) -> List[Chunk]:
        chunks = []
        for idx, passage in enumerate(passages):
            if not passage.strip():
                continue
            
            meta = metadata[idx] if metadata else {}
            
            # Enrich chunk with query context if available
            enriched_text = passage.strip()
            if meta.get("query"):
                enriched_text = f"[Query: {meta['query']}] {passage.strip()}"
            
            chunks.append(Chunk(
                text=enriched_text,
                strategy="metadata_aware",
                source_idx=idx,
                metadata={
                    **meta,
                    "is_selected": meta.get("is_selected", False),
                },
            ))
        return chunks


class SlidingWindowChunker(ChunkingStrategy):
    """
    Strategy 5: Sliding window with dynamic sizing.
    Adapts window size based on content density.
    """
    
    def __init__(self, base_size: int = 200, min_size: int = 100, max_size: int = 400, stride: int = 100):
        self.base_size = base_size
        self.min_size = min_size
        self.max_size = max_size
        self.stride = stride
    
    def _estimate_density(self, text: str) -> float:
        """Estimate information density (higher = more dense)."""
        words = text.split()
        if not words:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio
    
    def chunk(self, passages: List[str], metadata: Optional[List[dict]] = None) -> List[Chunk]:
        chunks = []
        for idx, passage in enumerate(passages):
            words = passage.split()
            if len(words) <= self.min_size:
                chunks.append(Chunk(
                    text=passage,
                    strategy="sliding_window",
                    source_idx=idx,
                    metadata=metadata[idx] if metadata else {},
                ))
                continue
            
            # Dynamic window size based on density
            density = self._estimate_density(passage)
            window_size = int(self.base_size * (1.0 + (1.0 - density)))
            window_size = max(self.min_size, min(self.max_size, window_size))
            
            start = 0
            while start < len(words):
                end = min(start + window_size, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append(Chunk(
                    text=chunk_text,
                    strategy="sliding_window",
                    source_idx=idx,
                    metadata={
                        **(metadata[idx] if metadata else {}),
                        "window_size": window_size,
                        "density": density,
                    },
                ))
                if end >= len(words):
                    break
                start += self.stride
        return chunks


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:
    """
    Core RAG pipeline with multi-strategy chunking and FAISS retrieval.
    
    Architecture:
    1. Load dataset → Extract passages with metadata
    2. Apply multiple chunking strategies
    3. Encode all chunks into embeddings
    4. Build FAISS index for fast retrieval
    5. At query time: encode query → search index → return top-k
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        index_path: Optional[str] = None,
    ):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Chunking strategies
        self.chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=256, overlap=50),
            "semantic_sentence": SemanticSentenceChunker(max_sentences=5),
            "passage_level": PassageLevelChunker(),
            "metadata_aware": MetadataAwareChunker(),
            "sliding_window": SlidingWindowChunker(base_size=200, stride=100),
        }
        
        # Storage
        self.chunks: List[Chunk] = []
        self.index: Optional[faiss.Index] = None
        self.is_ready = False
        
        # Load existing index if available
        if index_path and os.path.exists(index_path):
            self._load_index(index_path)
    
    def load_and_index_dataset(
        self,
        language: str = "hi",
        split: str = "train[:5000]",
        strategies: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        local_json_path: Optional[str] = None,
        max_samples: int = 5000,
    ):
        """
        Load dataset, apply chunking strategies, and build FAISS index.
        
        Supports:
        - Local JSON file (fastest, no download)
        - HuggingFace streaming (avoids full 3.7GB download)
        - HuggingFace direct loading (requires full download)
        """
        all_passages = []
        all_metadata = []
        
        # Method 1: Load from local JSON file
        if local_json_path and os.path.exists(local_json_path):
            print(f"Loading from local JSON: {local_json_path}")
            with open(local_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for example in data[:max_samples]:
                translated_passages = example["passages"]["Translated_passages"]
                is_selected = example["passages"]["is_selected"]
                query = example.get("query", "")
                
                for j, (passage, selected) in enumerate(zip(translated_passages, is_selected)):
                    if passage and passage.strip():
                        all_passages.append(passage)
                        all_metadata.append({
                            "query": query,
                            "is_selected": bool(selected),
                            "passage_index": j,
                        })
        else:
            # Method 2: Stream from HuggingFace (avoids full download)
            print(f"Streaming MSMARCO-XI dataset (lang={language})...")
            try:
                dataset = load_dataset(
                    "ai4bharat/MSMARCO-XI",
                    language,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"Config-based loading failed: {e}")
                print("Trying default config with streaming...")
                dataset = load_dataset(
                    "ai4bharat/MSMARCO-XI",
                    split="train",
                    streaming=True,
                )
            
            count = 0
            for example in dataset:
                if count >= max_samples:
                    break
                
                translated_passages = example["passages"]["Translated_passages"]
                is_selected = example["passages"]["is_selected"]
                query = example.get("query", "")
                
                for j, (passage, selected) in enumerate(zip(translated_passages, is_selected)):
                    if passage and passage.strip():
                        all_passages.append(passage)
                        all_metadata.append({
                            "query": query,
                            "is_selected": bool(selected),
                            "passage_index": j,
                        })
                
                count += 1
                if count % 500 == 0:
                    print(f"  Streamed {count} examples...")
        
        print(f"Extracted {len(all_passages)} passages from dataset")
        
        # Apply chunking strategies
        active_strategies = strategies or list(self.chunkers.keys())
        self.chunks = []
        
        for strategy_name in active_strategies:
            if strategy_name in self.chunkers:
                print(f"Applying chunking strategy: {strategy_name}")
                chunker = self.chunkers[strategy_name]
                strategy_chunks = chunker.chunk(all_passages, all_metadata)
                self.chunks.extend(strategy_chunks)
                print(f"  → {len(strategy_chunks)} chunks")
        
        print(f"Total chunks across all strategies: {len(self.chunks)}")
        
        # Deduplicate chunks by text (keep first occurrence)
        seen_texts = set()
        unique_chunks = []
        for chunk in self.chunks:
            text_key = chunk.text[:200]  # Use prefix for dedup
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_chunks.append(chunk)
        
        self.chunks = unique_chunks
        print(f"After deduplication: {len(self.chunks)} chunks")
        
        # Build embeddings
        print("Encoding chunks into embeddings...")
        texts = [c.text for c in self.chunks]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )
        
        # Build FAISS index (using IndexFlatIP for cosine similarity with normalized vectors)
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        
        self.is_ready = True
        print(f"RAG pipeline ready! Index contains {self.index.ntotal} vectors")
        
        # Save if path provided
        if save_path:
            self._save_index(save_path)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy_filter: Optional[str] = None,
    ) -> Tuple[List[RetrievedPassage], float]:
        """
        Retrieve top-k relevant passages for a query.
        Returns (passages, latency_ms).
        """
        if not self.is_ready:
            raise RuntimeError("RAG pipeline not initialized. Call load_and_index_dataset() first.")
        
        start_time = time.perf_counter()
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        
        # Search
        search_k = top_k * 3 if strategy_filter else top_k
        scores, indices = self.index.search(
            np.array(query_embedding, dtype=np.float32),
            min(search_k, self.index.ntotal),
        )
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Apply strategy filter if specified
            if strategy_filter and chunk.strategy != strategy_filter:
                continue
            
            results.append(RetrievedPassage(
                text=chunk.text,
                score=float(score),
                chunk_strategy=chunk.strategy,
                source_index=chunk.source_idx,
            ))
            
            if len(results) >= top_k:
                break
        
        latency = (time.perf_counter() - start_time) * 1000
        return results, latency
    
    def _save_index(self, path: str):
        """Save FAISS index and chunks to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save chunks metadata
        chunks_data = [
            {
                "text": c.text,
                "strategy": c.strategy,
                "source_idx": c.source_idx,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        with open(f"{path}.chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"Index saved to {path}")
    
    def _load_index(self, path: str):
        """Load FAISS index and chunks from disk."""
        index_file = f"{path}.faiss"
        chunks_file = f"{path}.chunks.json"
        
        if os.path.exists(index_file) and os.path.exists(chunks_file):
            self.index = faiss.read_index(index_file)
            
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            
            self.chunks = [
                Chunk(
                    text=c["text"],
                    strategy=c["strategy"],
                    source_idx=c["source_idx"],
                    metadata=c.get("metadata", {}),
                )
                for c in chunks_data
            ]
            
            self.is_ready = True
            print(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
