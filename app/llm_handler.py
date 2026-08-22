"""
LLM Handler for answer generation.

Uses Groq for fast inference with structured orchestration:
- Tool calls
- Retries
- Structured input/output
- Error recovery
"""

import os
import time
import json
import httpx
from typing import List, Optional

from app.models import RetrievedPassage


class LLMHandler:
    """
    LLM handler using Groq API for ultra-fast inference.
    
    Features:
    - Structured prompt templates
    - Retry with exponential backoff
    - Context window management
    - Structured JSON output parsing
    """
    
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        max_tokens: int = 100,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.fallback_api_key = os.getenv("GROQ_API_KEY_FALLBACK")
        self.model = model
        self.max_tokens = max_tokens
        
        if self.api_key:
            self.client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=0.2,
            )
        else:
            self.client = None
            
        if self.fallback_api_key:
            self.fallback_client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.fallback_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=0.2,
            )
        else:
            self.fallback_client = None
    
    def _build_prompt(
        self,
        query: str,
        passages: List[RetrievedPassage],
        language: str = "hi",
    ) -> List[dict]:
        """Build structured prompt with retrieved context (top 3 only for speed)."""
        
        # Only use top 3 passages to minimize token count and LLM latency
        top_passages = passages[:3]
        
        context_parts = []
        for i, passage in enumerate(top_passages, 1):
            # Trim each passage to max 100 chars to reduce tokens
            text = passage.text[:100]
            context_parts.append(f"[{i}] {text}")
        context = "\n".join(context_parts)
        
        system_prompt = f"""Answer ONLY from context. Reply strictly in the language of this ISO code: '{language}'. Be concise. Return JSON: {{"answer": "...", "sources_used": [1], "confidence": 0.8}}"""

        user_prompt = f"Context:\n{context}\n\nQ: {query}\nJSON:"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
    async def generate(
        self,
        query: str,
        passages: List[RetrievedPassage],
        language: str = "hi",
        max_retries: int = 2,
    ) -> dict:
        """
        Generate an answer using the LLM with retry logic and fallback client support.
        
        Returns: {"answer": str, "confidence": float, "sources_used": list, "latency_ms": float}
        """
        start_time = time.perf_counter()
        
        # Build client checklist
        clients_to_try = []
        if self.client:
            clients_to_try.append((self.client, "primary"))
        if self.fallback_client:
            clients_to_try.append((self.fallback_client, "fallback"))
            
        if not clients_to_try:
            return self._extractive_fallback(query, passages, start_time)
        
        messages = self._build_prompt(query, passages, language)
        last_error = None
        
        for client, name in clients_to_try:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        self.GROQ_API_URL,
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": self.max_tokens,
                            "temperature": 0.1,
                            "response_format": {"type": "json_object"},
                        },
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        # Parse structured JSON output
                        try:
                            parsed = json.loads(content)
                        except json.JSONDecodeError:
                            parsed = {"answer": content, "confidence": 0.5, "sources_used": []}
                        
                        latency = (time.perf_counter() - start_time) * 1000
                        
                        return {
                            "answer": parsed.get("answer", content),
                            "confidence": parsed.get("confidence", 0.5),
                            "sources_used": parsed.get("sources_used", []),
                            "latency_ms": latency,
                        }
                    
                    elif response.status_code == 429:
                        # Rate limited, back off and retry
                        import asyncio
                        await asyncio.sleep(2 ** attempt * 0.5)
                    else:
                        last_error = f"LLM API error ({name}): {response.status_code} - {response.text}"
                        # If auth error or model not found, switch to fallback key immediately
                        if response.status_code in (401, 403, 404):
                            break
                        
                except Exception as e:
                    last_error = f"LLM error ({name}): {str(e)}"
                    import asyncio
                    await asyncio.sleep(2 ** attempt * 0.5)
        
        # All clients and retries failed, use extractive fallback
        print(f"LLM generation failed for all clients. Last error: {last_error}. Using extractive fallback.")
        return self._extractive_fallback(query, passages, start_time)
    
    def _extractive_fallback(
        self,
        query: str,
        passages: List[RetrievedPassage],
        start_time: float,
    ) -> dict:
        """
        Extractive fallback when LLM is unavailable.
        Returns the best matching passage as the answer.
        """
        if not passages:
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "answer": "No relevant information found.",
                "confidence": 0.0,
                "sources_used": [],
                "latency_ms": latency,
            }
        
        # Use the top passage as the answer
        best_passage = passages[0]
        
        # Trim to a reasonable answer length
        answer_text = best_passage.text
        if len(answer_text) > 500:
            # Find the best sentence that relates to the query
            sentences = answer_text.replace("।", ".").split(".")
            answer_text = ". ".join(sentences[:3]) + "."
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            "answer": answer_text,
            "confidence": min(best_passage.score, 1.0),
            "sources_used": [1],
            "latency_ms": latency,
        }
    
    async def close(self):
        """Close HTTP clients."""
        if self.client:
            await self.client.aclose()
        if hasattr(self, "fallback_client") and self.fallback_client:
            await self.fallback_client.aclose()
