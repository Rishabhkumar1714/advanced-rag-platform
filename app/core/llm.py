"""
LLM Client — Groq API (Free)
Uses Groq's free API with llama3-8b-8192 for ultra-fast inference.
"""

import time
from typing import AsyncGenerator, Dict, List, Optional

from groq import AsyncGroq

from app.config import settings
from app.models.query import SourceDocument
from app.utils.helpers import count_tokens, truncate_to_token_limit
import logging

logger = logging.getLogger(__name__)

# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert knowledge assistant powered by a Retrieval-Augmented Generation (RAG) system.

Your role is to provide accurate, well-reasoned answers grounded strictly in the retrieved context provided to you.

Guidelines:
- Base your answer solely on the provided context. Do NOT use external knowledge.
- If the context does not contain enough information to answer the question, say so clearly.
- Cite the source documents when relevant by referring to their titles.
- Keep answers concise but thorough. Use bullet points or numbered lists when they improve clarity.
- Never fabricate facts, statistics, or references not present in the context.
"""

CONTEXT_TEMPLATE = """<context>
{context_blocks}
</context>

Based on the context above, please answer the following question:
{question}"""


def build_context_prompt(
    question: str,
    sources: List[SourceDocument],
    max_tokens: int = settings.max_context_tokens,
) -> str:
    context_parts: List[str] = []
    used_tokens = count_tokens(CONTEXT_TEMPLATE.format(context_blocks="", question=question))

    for i, source in enumerate(sources, start=1):
        block = (
            f"[Source {i}] Title: {source.title}\n"
            f"Relevance Score: {source.score:.2f}\n"
            f"Content: {source.content}"
        )
        block_tokens = count_tokens(block)
        if used_tokens + block_tokens > max_tokens:
            remaining = max_tokens - used_tokens
            if remaining > 50:
                block = truncate_to_token_limit(block, remaining)
                context_parts.append(block)
            break
        context_parts.append(block)
        used_tokens += block_tokens

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."
    return CONTEXT_TEMPLATE.format(context_blocks=context_str, question=question)


# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(
        self,
        model: str = settings.groq_model,
        max_tokens: int = settings.groq_max_tokens,
        temperature: float = settings.groq_temperature,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    def _build_messages(
        self,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-6:])
        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def generate(
        self,
        question: str,
        sources: List[SourceDocument],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        """Generate a non-streaming RAG response via Groq."""
        user_prompt = build_context_prompt(question, sources)
        messages = self._build_messages(user_prompt, conversation_history)

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        answer = response.choices[0].message.content or ""
        usage = response.usage

        logger.info(
            "Groq LLM response generated",
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=round(latency_ms, 2),
        )

        return {
            "answer": answer,
            "model_used": self.model,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "latency_ms": round(latency_ms, 2),
        }

    async def stream(
        self,
        question: str,
        sources: List[SourceDocument],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream RAG response tokens via Groq."""
        user_prompt = build_context_prompt(question, sources)
        messages = self._build_messages(user_prompt, conversation_history)

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
