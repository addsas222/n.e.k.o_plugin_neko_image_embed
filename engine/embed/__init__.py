"""嵌入后端：CLIP ONNX（首选）/ 外部 embeddings（备选）。"""

from __future__ import annotations

from .clip_onnx import ClipEmbedder, ClipError
from .external import ExternalEmbedder, ExternalError
from .registry import EmbeddingRegistry, EmbedError, EmbedState

__all__ = [
    "ClipEmbedder",
    "ClipError",
    "EmbedError",
    "EmbedState",
    "EmbeddingRegistry",
    "ExternalEmbedder",
    "ExternalError",
]
