"""
LLM Client — Groq API (Free - llama-3.1-8b-instant)
"""

import logging
import os
import time
from typing import AsyncGenerator, Dict, List, Optional

from app.models.query import SourceDocument
from app.utils.helpers import count_tokens, truncate_to_token_limit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert knowledge assistant powered by a RAG system.
Answer questions based on the retrieved context provided.
If context is empty or insufficient, say: "I don't have enough information in the knowledge base to answer this. Please add relevant documents first."
Be concise, accurate, and helpful."""

CONTEXT_TEMPLATE = """<context>
{context_blocks}
</context>

Question: {question}

Answer based only on the context above:"""


def build_context_prompt(question: str, sources: List[SourceDocument], max_tokens: int = 3000) -> str:
    if not sources:
        return f"No relevant context found in knowledge base.\n\nQuestion: {question}"
    context_parts = []
    for i, source in enumerate(sources, start=1):
        block = f"[Source {i}] {source.title}\n{source.content}"
        context_parts.append(block)
    context_str = "\n\n---\n\n".join(context_parts)
    return CONTEXT_TEMPLATE.format(context_blocks=context_str, question=question)


class LLMClient:
    def __init__(self):
        # Read directly from environment to be safe
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        self.max_tokens = 1024
        self.temperature = 0.1
        self._client = None

        if not self.api_key:
            logger.error("GROQ_API_KEY is not set! Queries will fail.")
        else:
            logger.info(f"Groq client configured: model={self.model}, key=...{self.api_key[-4:]}")

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self.api_key)
        return self._client

    async def generate(self, question: str, sources: List[SourceDocument], conversation_history=None) -> Dict:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set. Please add it in Render → Environment Variables.")

        user_prompt = build_context_prompt(question, sources)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if conversation_history:
            messages = [messages[0]] + conversation_history[-4:] + [messages[-1]]

        client = self._get_client()
        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(f"Groq API call failed: {type(e).__name__}: {e}")
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        answer = response.choices[0].message.content or ""
        usage = response.usage
        logger.info(f"Groq OK: tokens={getattr(usage,'total_tokens',0)}, latency={latency_ms:.0f}ms")
        return {
            "answer": answer,
            "model_used": self.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "latency_ms": round(latency_ms, 2),
        }

    async def stream(self, question: str, sources: List[SourceDocument], conversation_history=None) -> AsyncGenerator[str, None]:
        if not self.api_key:
            yield "Error: GROQ_API_KEY is not set."
            return
        user_prompt = build_context_prompt(question, sources)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=self.max_tokens, temperature=self.temperature, stream=True,
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
