"""
LLM Client
Async OpenAI wrapper with context-aware prompt construction,
conversation history management, and streaming support.
"""

import time
from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.config import settings
from app.models.query import SourceDocument
from app.utils.helpers import count_tokens, truncate_to_token_limit
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert knowledge assistant powered by a Retrieval-Augmented Generation (RAG) system.

Your role is to provide accurate, well-reasoned answers grounded strictly in the retrieved context provided to you.

Guidelines:
- Base your answer solely on the provided context. Do NOT use external knowledge.
- If the context does not contain enough information to answer the question, say so clearly.
- Cite the source documents when relevant by referring to their titles.
- Keep answers concise but thorough. Use bullet points or numbered lists when they improve clarity.
- Never fabricate facts, statistics, or references not present in the context.
- If the question is ambiguous, ask for clarification before answering.
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
    """
    Build the user turn of the prompt by assembling retrieved chunks
    into a context block, respecting the token budget.
    """
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
            # Truncate the last block to fit
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
        model: str = settings.openai_model,
        max_tokens: int = settings.openai_max_tokens,
        temperature: float = settings.openai_temperature,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    def _build_messages(
        self,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Keep last 3 turns
        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def generate(
        self,
        question: str,
        sources: List[SourceDocument],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        """Generate a non-streaming RAG response."""
        user_prompt = build_context_prompt(question, sources)
        messages = self._build_messages(user_prompt, conversation_history)

        start = time.perf_counter()
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        answer = response.choices[0].message.content or ""
        usage = response.usage

        logger.info(
            "LLM response generated",
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
        """Generate a streaming RAG response, yielding token chunks."""
        user_prompt = build_context_prompt(question, sources)
        messages = self._build_messages(user_prompt, conversation_history)

        async with await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        ) as stream:
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
