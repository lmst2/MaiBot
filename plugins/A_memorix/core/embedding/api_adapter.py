"""
Hash-based embedding adapter used by the SDK runtime.

The plugin runtime cannot import MaiBot host embedding internals from ``src.chat``
or ``src.llm_models``. This adapter keeps A_Memorix self-contained and stable in
Runner by generating deterministic dense vectors locally.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import List, Optional, Union

import numpy as np

from src.common.logger import get_logger


logger = get_logger("A_Memorix.EmbeddingAPIAdapter")

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]{1,}")


class EmbeddingAPIAdapter:
    """Deterministic local embedding adapter."""

    def __init__(
        self,
        batch_size: int = 32,
        max_concurrent: int = 5,
        default_dimension: int = 256,
        enable_cache: bool = False,
        model_name: str = "hash-v1",
        retry_config: Optional[dict] = None,
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.max_concurrent = max(1, int(max_concurrent))
        self.default_dimension = max(32, int(default_dimension))
        self.enable_cache = bool(enable_cache)
        self.model_name = str(model_name or "hash-v1")
        self.retry_config = retry_config or {}

        self._dimension: Optional[int] = None
        self._dimension_detected = False
        self._total_encoded = 0
        self._total_errors = 0
        self._total_time = 0.0

        logger.info(
            "EmbeddingAPIAdapter 初始化: model=%s, batch_size=%s, dimension=%s",
            self.model_name,
            self.batch_size,
            self.default_dimension,
        )

    async def _detect_dimension(self) -> int:
        if self._dimension_detected and self._dimension is not None:
            return self._dimension
        self._dimension = self.default_dimension
        self._dimension_detected = True
        return self._dimension

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        clean = str(text or "").strip().lower()
        if not clean:
            return []
        return _TOKEN_PATTERN.findall(clean)

    @staticmethod
    def _feature_weight(token: str) -> float:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return 1.0 + (digest[10] / 255.0) * 0.5

    def _encode_single(self, text: str, dimension: int) -> np.ndarray:
        vector = np.zeros(dimension, dtype=np.float32)
        content = str(text or "").strip()
        tokens = self._tokenize(content)
        if not tokens and content:
            tokens = [content.lower()]
        if not tokens:
            vector[0] = 1.0
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[bucket] += sign * self._feature_weight(token)

            second_bucket = int.from_bytes(digest[12:20], byteorder="big", signed=False) % dimension
            if second_bucket != bucket:
                vector[second_bucket] += (sign * 0.35)

        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            vector /= norm
        else:
            vector[0] = 1.0
        return vector

    async def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        _ = batch_size
        _ = show_progress
        _ = normalize

        started_at = time.time()
        target_dimension = max(32, int(dimensions or await self._detect_dimension()))

        if isinstance(texts, str):
            single_input = True
            normalized_texts = [texts]
        else:
            single_input = False
            normalized_texts = list(texts or [])

        if not normalized_texts:
            empty = np.zeros((0, target_dimension), dtype=np.float32)
            return empty[0] if single_input else empty

        try:
            matrix = np.vstack([self._encode_single(item, target_dimension) for item in normalized_texts])
            self._total_encoded += len(normalized_texts)
            self._total_time += time.time() - started_at
        except Exception:
            self._total_errors += 1
            raise

        return matrix[0] if single_input else matrix

    def get_statistics(self) -> dict:
        avg_time = self._total_time / self._total_encoded if self._total_encoded else 0.0
        return {
            "model_name": self.model_name,
            "dimension": self._dimension or self.default_dimension,
            "total_encoded": self._total_encoded,
            "total_errors": self._total_errors,
            "total_time": self._total_time,
            "avg_time_per_text": avg_time,
        }

    def __repr__(self) -> str:
        return (
            f"EmbeddingAPIAdapter(model_name={self.model_name}, "
            f"dimension={self._dimension or self.default_dimension}, "
            f"total_encoded={self._total_encoded})"
        )


def create_embedding_api_adapter(
    batch_size: int = 32,
    max_concurrent: int = 5,
    default_dimension: int = 256,
    enable_cache: bool = False,
    model_name: str = "hash-v1",
    retry_config: Optional[dict] = None,
) -> EmbeddingAPIAdapter:
    return EmbeddingAPIAdapter(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        default_dimension=default_dimension,
        enable_cache=enable_cache,
        model_name=model_name,
        retry_config=retry_config,
    )
