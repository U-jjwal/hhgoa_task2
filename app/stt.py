"""Speech-to-Text integration using Sarvam AI."""

import os
import time
import base64
import httpx
from typing import Optional

from app.models import STTResponse


class SarvamSTT:
    """Sarvam AI Speech-to-Text client with retry logic."""
    
    BASE_URL = "https://api.sarvam.ai"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY not set. Get one at https://sarvam.ai/")
        
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "api-subscription-key": self.api_key,
            },
            timeout=30.0,
        )
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "hi-IN",
        max_retries: int = 3,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> STTResponse:
        """
        Transcribe audio bytes to text using Sarvam STT API.
        
        Implements retry logic for robustness.
        """
        start_time = time.perf_counter()
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Sarvam expects multipart form data with the audio file
                response = await self.client.post(
                    "/speech-to-text",
                    files={
                        "file": (filename or "audio.wav", audio_bytes, content_type or "audio/wav"),
                    },
                    data={
                        "language_code": language_code,
                        "model": "saarika:v2.5",
                    },
                )
                
                if response.status_code == 200:
                    result = response.json()
                    latency = (time.perf_counter() - start_time) * 1000
                    
                    return STTResponse(
                        transcript=result.get("transcript", ""),
                        language_detected=result.get("language_code", language_code),
                        confidence=result.get("confidence", 0.9),
                        latency_ms=latency,
                    )
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    await self._backoff(attempt)
                else:
                    last_error = f"STT API error: {response.status_code} - {response.text}"
                    
            except httpx.TimeoutException:
                last_error = f"STT timeout on attempt {attempt + 1}"
                await self._backoff(attempt)
            except Exception as e:
                last_error = f"STT error: {str(e)}"
                await self._backoff(attempt)
        
        # All retries failed
        raise RuntimeError(f"STT failed after {max_retries} retries. Last error: {last_error}")
    
    async def _backoff(self, attempt: int):
        """Exponential backoff between retries."""
        import asyncio
        wait_time = min(2 ** attempt * 0.5, 5.0)
        await asyncio.sleep(wait_time)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class MockSTT:
    """Mock STT for testing without API access."""
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "hi-IN",
        max_retries: int = 3,
    ) -> STTResponse:
        """Return a mock transcription for testing."""
        start_time = time.perf_counter()
        
        # Simulate some processing time
        import asyncio
        await asyncio.sleep(0.05)
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return STTResponse(
            transcript="मेनहट्टन प्रोजेक्ट क्या था?",
            language_detected="hi-IN",
            confidence=0.95,
            latency_ms=latency,
        )
    
    async def close(self):
        pass
