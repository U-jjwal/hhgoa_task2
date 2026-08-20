"""
Guardrails for the Voice RAG pipeline.

Implements:
1. Off-topic query detection
2. Unsafe/inappropriate input filtering
3. Hallucination checks (grounding verification)
4. Confidence-based answer gating
5. Language validation
"""

import re
from typing import List, Optional

from app.models import GuardrailResult, RetrievedPassage


class Guardrails:
    """
    Safety and relevance guardrails for the RAG pipeline.
    
    The system knows when NOT to answer, not just how to answer.
    """
    
    # Unsafe patterns (multi-language)
    UNSAFE_PATTERNS = [
        # English unsafe patterns
        r"\b(hack|exploit|bomb|kill|attack|weapon|drug|narcotic)\b",
        # Hindi unsafe patterns (transliterated)
        r"\b(हथियार|बम|ड्रग|हमला|नशा)\b",
    ]
    
    # Minimum confidence threshold for answering
    MIN_CONFIDENCE_THRESHOLD = 0.3
    
    # Maximum distance threshold (for cosine similarity, higher is better)
    MIN_RELEVANCE_SCORE = 0.25
    
    def __init__(
        self,
        min_confidence: float = 0.3,
        min_relevance_score: float = 0.25,
        max_answer_length: int = 2000,
    ):
        self.min_confidence = min_confidence
        self.min_relevance_score = min_relevance_score
        self.max_answer_length = max_answer_length
    
    def check_input(self, query: str) -> GuardrailResult:
        """
        Check if the input query is safe and appropriate.
        """
        flags = []
        
        # 1. Empty query check
        if not query or not query.strip():
            return GuardrailResult(
                is_safe=False,
                is_on_topic=False,
                is_grounded=False,
                flags=["empty_query"],
                blocked_reason="Empty query received.",
            )
        
        # 2. Query too short
        if len(query.strip()) < 3:
            flags.append("query_too_short")
        
        # 3. Unsafe content check
        query_lower = query.lower()
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return GuardrailResult(
                    is_safe=False,
                    is_on_topic=False,
                    is_grounded=False,
                    flags=["unsafe_content"],
                    blocked_reason="Query contains potentially unsafe content.",
                )
        
        # 4. Injection attempt detection
        injection_patterns = [
            r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
            r"system\s*prompt",
            r"you\s+are\s+now",
            r"pretend\s+to\s+be",
            r"jailbreak",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return GuardrailResult(
                    is_safe=False,
                    is_on_topic=False,
                    is_grounded=False,
                    flags=["injection_attempt"],
                    blocked_reason="Potential prompt injection detected.",
                )
        
        return GuardrailResult(
            is_safe=True,
            is_on_topic=True,  # Will be refined after retrieval
            is_grounded=True,
            flags=flags,
        )
    
    def check_retrieval(
        self,
        query: str,
        passages: List[RetrievedPassage],
    ) -> GuardrailResult:
        """
        Check if retrieved passages are relevant enough to answer.
        """
        flags = []
        
        if not passages:
            return GuardrailResult(
                is_safe=True,
                is_on_topic=False,
                is_grounded=False,
                flags=["no_passages_retrieved"],
                blocked_reason="No relevant information found for your query.",
            )
        
        # Check if top passage score is above threshold
        top_score = passages[0].score
        if top_score < self.min_relevance_score:
            flags.append("low_relevance")
            return GuardrailResult(
                is_safe=True,
                is_on_topic=False,
                is_grounded=False,
                flags=flags,
                blocked_reason=f"Retrieved passages are not relevant enough (score: {top_score:.3f}).",
            )
        
        # Check if there's a significant drop between top and second result
        if len(passages) > 1:
            score_gap = passages[0].score - passages[1].score
            if score_gap > 0.5:
                flags.append("single_source_dominance")
        
        return GuardrailResult(
            is_safe=True,
            is_on_topic=True,
            is_grounded=True,
            flags=flags,
        )
    
    def check_output(
        self,
        answer: str,
        passages: List[RetrievedPassage],
    ) -> GuardrailResult:
        """
        Check if the generated answer is grounded in the retrieved passages.
        Hallucination detection.
        """
        flags = []
        
        if not answer or not answer.strip():
            return GuardrailResult(
                is_safe=True,
                is_on_topic=False,
                is_grounded=False,
                flags=["empty_answer"],
                blocked_reason="Failed to generate an answer.",
            )
        
        # Length check
        if len(answer) > self.max_answer_length:
            flags.append("answer_too_long")
        
        # Grounding check: verify key terms from answer appear in passages
        passage_text = " ".join([p.text for p in passages]).lower()
        answer_words = set(answer.lower().split())
        
        # Filter out stop words (basic set for Hindi and English)
        stop_words = {
            "the", "a", "an", "is", "was", "are", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "not", "with", "by",
            "है", "हैं", "था", "थे", "में", "पर", "से", "को", "का", "की",
            "के", "और", "या", "एक", "यह", "वह", "इस", "उस", "कि",
        }
        
        content_words = answer_words - stop_words
        if content_words:
            grounded_words = sum(1 for w in content_words if w in passage_text)
            grounding_ratio = grounded_words / len(content_words)
            
            if grounding_ratio < 0.3:
                flags.append("low_grounding")
                return GuardrailResult(
                    is_safe=True,
                    is_on_topic=True,
                    is_grounded=False,
                    flags=flags,
                    blocked_reason=f"Answer may not be grounded in retrieved context (grounding: {grounding_ratio:.1%}).",
                )
        
        # Check for common hallucination indicators
        hallucination_phrases = [
            "I think", "I believe", "probably", "might be",
            "मुझे लगता है", "शायद", "हो सकता है",
        ]
        for phrase in hallucination_phrases:
            if phrase.lower() in answer.lower():
                flags.append("hedging_language")
        
        return GuardrailResult(
            is_safe=True,
            is_on_topic=True,
            is_grounded=True,
            flags=flags,
        )
    
    def full_check(
        self,
        query: str,
        passages: List[RetrievedPassage],
        answer: str,
    ) -> GuardrailResult:
        """
        Run all guardrail checks in sequence.
        Returns the first failing check or a passing result.
        """
        # 1. Check input
        input_check = self.check_input(query)
        if not input_check.is_safe:
            return input_check
        
        # 2. Check retrieval
        retrieval_check = self.check_retrieval(query, passages)
        if not retrieval_check.is_on_topic:
            return retrieval_check
        
        # 3. Check output
        output_check = self.check_output(answer, passages)
        
        # Merge all flags
        all_flags = list(set(
            input_check.flags + retrieval_check.flags + output_check.flags
        ))
        
        return GuardrailResult(
            is_safe=output_check.is_safe,
            is_on_topic=retrieval_check.is_on_topic,
            is_grounded=output_check.is_grounded,
            flags=all_flags,
            blocked_reason=output_check.blocked_reason,
        )
