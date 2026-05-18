"""
LLM Client — Groq API (Free, llama3-8b-8192)
"""

import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

from groq import AsyncGroq

from app.config import settings
from app.models.query import SourceDocument
from app.utils.helpers import count_tokens, truncate_to_token_limit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert knowledge assistant powered by a Retrieval-Augmented Generation (RAG) system.
Answer questions based solely on the retrieved context provided. If the context doesn't contain enough information, say so clearly.
Be concise, accurate, and cite source titles when relevant."""

CONTEXT_TEMPLATE = """<context>
{context_blocks}
</context>

Question: {question}"""


def build_context_prompt(question: str, sources: List[SourceDocument], max_tokens: int = settings.max_context_tokens) -> str:
    context_parts: List[str] = []
    used_tokens = count_tokens(CONTEXT_TEMPLATE.format(context_blocks="", question=question))
    for i, source in enumerate(sources, start=1):
        block = f"[Source {i}] Title: {source.title}\nScore: {source.score:.2f}\nContent: {source.content}"
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


class LLMClient:
    def __init__(self, model: str = settings.groq_model, max_tokens: int = settings.groq_max_tokens, temperature: float = settings.groq_temperature):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    def _build_messages(self, user_prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history[-6:])
        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def generate(self, question: str, sources: List[SourceDocument], conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict:
        user_prompt = build_context_prompt(question, sources)
        messages = self._build_messages(user_prompt, conversation_history)
        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=self.temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        answer = response.choices[0].message.content or ""
        usage = response.usage
        logger.info(f"Groq response: tokens={getattr(usage, 'total_tokens', 0)}, latency={latency_ms:.0f}ms")
        return {
            "answer": answer, "model_used": self.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "latency_ms": round(latency_ms, 2),
        }

    async def stream(self, question: str, sources: List[SourceDocument], conversation_history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[str, None]:
        user_prompt = build_context_prompt(question, sources)
        messages = self._build_messages(user_prompt, conversation_history)
        stream = await self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=self.temperature, stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


_llm_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
