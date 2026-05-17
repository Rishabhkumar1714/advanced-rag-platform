"""
Intelligent Text Chunking
Implements multiple chunking strategies: recursive character splitting,
semantic sentence-boundary splitting, and sliding-window chunking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import settings
from app.utils.helpers import count_tokens
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        ...


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators,
    respecting paragraph → sentence → word boundaries.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        separator = separators[0]
        remaining = separators[1:]

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []
        current = ""

        for s in splits:
            candidate = f"{current}{separator}{s}" if current else s
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good_splits.append(current)
                if len(s) > self.chunk_size:
                    good_splits.extend(self._split_text(s, remaining))
                    current = ""
                else:
                    current = s

        if current:
            good_splits.append(current)

        return good_splits

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        metadata = metadata or {}
        raw_splits = self._split_text(text, self.SEPARATORS)

        chunks: List[TextChunk] = []
        buffer = ""
        char_pos = 0

        for split in raw_splits:
            candidate = f"{buffer} {split}".strip() if buffer else split
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    start = text.find(buffer, char_pos)
                    end = start + len(buffer)
                    chunks.append(
                        TextChunk(
                            content=buffer,
                            chunk_index=len(chunks),
                            start_char=max(0, start),
                            end_char=end,
                            token_count=count_tokens(buffer),
                            metadata={**metadata, "strategy": "recursive_character"},
                        )
                    )
                    char_pos = max(0, end - self.chunk_overlap)
                buffer = split

        if buffer:
            start = text.find(buffer, char_pos)
            end = start + len(buffer)
            chunks.append(
                TextChunk(
                    content=buffer,
                    chunk_index=len(chunks),
                    start_char=max(0, start),
                    end_char=end,
                    token_count=count_tokens(buffer),
                    metadata={**metadata, "strategy": "recursive_character"},
                )
            )

        logger.info(
            "Chunking complete",
            strategy="recursive_character",
            chunk_count=len(chunks),
            text_length=len(text),
        )
        return chunks


class SemanticChunker(BaseChunker):
    """
    Sentence-aware chunker that groups sentences into chunks
    respecting a max token budget.
    """

    def __init__(
        self,
        max_tokens: int = settings.chunk_size,
        overlap_sentences: int = 2,
    ):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        metadata = metadata or {}
        sentences = self._split_sentences(text)

        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        current_tokens = 0
        char_offset = 0

        for sentence in sentences:
            s_tokens = count_tokens(sentence)
            if current_tokens + s_tokens > self.max_tokens and current_sentences:
                content = " ".join(current_sentences)
                start = text.find(content, char_offset)
                chunks.append(
                    TextChunk(
                        content=content,
                        chunk_index=len(chunks),
                        start_char=max(0, start),
                        end_char=max(0, start) + len(content),
                        token_count=current_tokens,
                        metadata={**metadata, "strategy": "semantic"},
                    )
                )
                # Overlap: keep last N sentences
                current_sentences = current_sentences[-self.overlap_sentences :]
                current_tokens = sum(count_tokens(s) for s in current_sentences)
                char_offset = max(0, start + len(content) - sum(len(s) for s in current_sentences))

            current_sentences.append(sentence)
            current_tokens += s_tokens

        if current_sentences:
            content = " ".join(current_sentences)
            start = text.find(content, char_offset)
            chunks.append(
                TextChunk(
                    content=content,
                    chunk_index=len(chunks),
                    start_char=max(0, start),
                    end_char=max(0, start) + len(content),
                    token_count=current_tokens,
                    metadata={**metadata, "strategy": "semantic"},
                )
            )

        logger.info(
            "Chunking complete",
            strategy="semantic",
            chunk_count=len(chunks),
            sentence_count=len(sentences),
        )
        return chunks


class SlidingWindowChunker(BaseChunker):
    """Fixed-size sliding window chunker with configurable overlap."""

    def __init__(
        self,
        window_size: int = settings.chunk_size,
        step_size: Optional[int] = None,
    ):
        self.window_size = window_size
        self.step_size = step_size or max(1, window_size - settings.chunk_overlap)

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        metadata = metadata or {}
        chunks: List[TextChunk] = []
        start = 0

        while start < len(text):
            end = min(start + self.window_size, len(text))
            content = text[start:end]
            chunks.append(
                TextChunk(
                    content=content,
                    chunk_index=len(chunks),
                    start_char=start,
                    end_char=end,
                    token_count=count_tokens(content),
                    metadata={**metadata, "strategy": "sliding_window"},
                )
            )
            if end == len(text):
                break
            start += self.step_size

        logger.info(
            "Chunking complete",
            strategy="sliding_window",
            chunk_count=len(chunks),
        )
        return chunks


class ChunkerFactory:
    """Factory to instantiate a chunker by strategy name."""

    _registry = {
        "recursive": RecursiveCharacterChunker,
        "semantic": SemanticChunker,
        "sliding_window": SlidingWindowChunker,
    }

    @classmethod
    def create(cls, strategy: str = "recursive", **kwargs) -> BaseChunker:
        if strategy not in cls._registry:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Choose from: {list(cls._registry)}"
            )
        return cls._registry[strategy](**kwargs)
