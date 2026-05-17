"""
General-purpose helper utilities.
"""

import hashlib
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import tiktoken


def generate_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def generate_document_id(content: str, source: str) -> str:
    """Generate a deterministic document ID from content + source."""
    payload = f"{source}::{content[:500]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate text to a maximum token count."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def sanitize_filename(filename: str) -> str:
    """Remove or replace characters unsafe for filenames."""
    filename = re.sub(r'[^\w\s\-.]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename[:255]


def get_file_extension(filename: str) -> str:
    """Return the lowercased file extension including the dot."""
    return Path(filename).suffix.lower()


def chunk_list(lst: List[Any], chunk_size: int) -> Generator[List[Any], None, None]:
    """Yield successive chunks of a list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def flatten_metadata(metadata: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested metadata dicts for vector store compatibility."""
    result: Dict[str, Any] = {}
    for key, value in metadata.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_metadata(value, prefix=full_key))
        elif isinstance(value, list):
            result[full_key] = str(value)
        else:
            result[full_key] = value
    return result


def timeit(func):
    """Decorator to measure and log execution time of a function."""
    import functools
    import asyncio
    from app.utils.logger import get_logger

    logger = get_logger("perf")

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"{func.__qualname__} completed", duration_ms=round(elapsed, 2))
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"{func.__qualname__} completed", duration_ms=round(elapsed, 2))
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def ensure_directory(path: str) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
