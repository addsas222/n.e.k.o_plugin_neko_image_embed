"""编码器注册表：按可用性选择后端并如实报告降级原因。

三级（对应配置里的 backend）：

1. ``clip_onnx``    —— 本地 CLIP 视觉塔，首选；
2. ``caption_text`` —— 视觉塔不可用、但文本塔可用时，用判定链的 caption 编码，
   与 1 共享同一个向量空间；
3. ``external``     —— 外部 ``/embeddings``，独立空间，换它必须重建索引。
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import EmbedderConfig
from .clip_onnx import ClipEmbedder, ClipError
from .external import ExternalEmbedder, ExternalError
from .fetch import ensure_model


class EmbedError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbedState:
    backend: str
    ready: bool
    detail: str
    id: str = ""
    dim: int = 0


class EmbeddingRegistry:
    def __init__(
        self,
        cfg: EmbedderConfig,
        *,
        data_dir: Path,
        plugin_dir: Path,
        timeout: float = 60.0,
    ) -> None:
        self._cfg = cfg
        self._data_dir = Path(data_dir)
        self._plugin_dir = Path(plugin_dir)
        self._timeout = timeout
        self._clip: ClipEmbedder | None = None
        self._external: ExternalEmbedder | None = None
        self._state = EmbedState("none", False, "尚未初始化")
        self.last_mode = ""

    @property
    def state(self) -> EmbedState:
        return self._state

    @property
    def id(self) -> str:
        return self._state.id

    @property
    def dim(self) -> int:
        return self._state.dim

    def model_dir(self) -> Path:
        if self._cfg.model_dir:
            return Path(self._cfg.model_dir)
        return self._data_dir / "models" / "clip-vit-base-patch32"

    async def prepare(self) -> EmbedState:
        self._state = await asyncio.to_thread(self._prepare_sync)
        return self._state

    def _prepare_sync(self) -> EmbedState:
        backend = self._cfg.backend
        detail = ""

        if backend in {"auto", "clip_onnx", "caption_text"}:
            model_dir = self.model_dir()
            model_state = ensure_model(
                model_dir,
                lock_path=self._plugin_dir / "models.lock.json",
                endpoint=self._cfg.hf_endpoint,
                allow_download=self._cfg.allow_download,
                timeout=self._timeout,
            )
            if model_state.ok:
                try:
                    clip = ClipEmbedder(model_dir)
                    clip.load()
                    self._clip = clip
                    chosen = "caption_text" if backend == "caption_text" else "clip_onnx"
                    return EmbedState(
                        backend=chosen,
                        ready=True,
                        detail=model_state.detail + ("（本次下载）" if model_state.downloaded else ""),
                        id=clip.id,
                        dim=clip.dim,
                    )
                except ClipError as exc:
                    detail = f"CLIP 加载失败: {exc}"
            else:
                detail = model_state.detail

        if backend in {"auto", "external"}:
            external = ExternalEmbedder(self._cfg, timeout=self._timeout)
            ok, why = external.configured()
            if ok:
                try:
                    probe = external.embed("dimension probe")
                except ExternalError as exc:
                    detail = (detail + "；" if detail else "") + f"外部 embeddings 探测失败: {exc}"
                else:
                    self._external = external
                    return EmbedState(
                        backend="external",
                        ready=True,
                        detail=(detail + "；" if detail else "") + "已回落到外部 embeddings",
                        id=external.id,
                        dim=int(probe.size),
                    )
            else:
                detail = (detail + "；" if detail else "") + f"外部 embeddings 未配置（{why}）"

        return EmbedState("none", False, detail or "没有可用的编码器")

    # ------------------------------------------------------------------ 编码

    async def embed_image(self, jpeg: bytes, *, caption: str = "") -> tuple[np.ndarray, str]:
        return await asyncio.to_thread(self._embed_image_sync, jpeg, caption)

    def _embed_image_sync(self, jpeg: bytes, caption: str) -> tuple[np.ndarray, str]:
        backend = self._state.backend
        if self._clip is not None and backend != "caption_text":
            try:
                with Image.open(io.BytesIO(jpeg)) as image:
                    vector = self._clip.embed_image(image.convert("RGB"))
                self.last_mode = "vision"
                return vector, "vision"
            except (ClipError, OSError, ValueError):
                # 视觉塔出问题时，用 caption 走同一文本塔，空间不变
                pass

        if self._clip is not None and caption.strip():
            try:
                vector = self._clip.embed_text(caption)
                self.last_mode = "caption"
                return vector, "caption"
            except ClipError:
                pass

        if self._external is not None and caption.strip():
            vector = self._external.embed(caption)
            self.last_mode = "external"
            return vector, "external"

        raise EmbedError(f"图像编码失败（后端 {backend or 'none'}，caption {len(caption)} 字）")

    async def embed_text(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._embed_text_sync, text)

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """批量文本编码；CLIP 不可用时退回逐条（外部 embeddings 没有批量语义）。"""
        return await asyncio.to_thread(self._embed_texts_sync, texts)

    def _embed_texts_sync(self, texts: list[str]) -> np.ndarray:
        if self._clip is not None:
            try:
                return self._clip.embed_texts(list(texts))
            except ClipError as exc:
                if self._external is None:
                    raise EmbedError(f"批量文本编码失败: {exc}") from exc
        return np.vstack([self._embed_text_sync(text) for text in texts])

    def _embed_text_sync(self, text: str) -> np.ndarray:
        cleaned = (text or "").strip()
        if not cleaned:
            raise EmbedError("查询文本为空")
        if self._clip is not None:
            try:
                return self._clip.embed_text(cleaned)
            except ClipError as exc:
                if self._external is None:
                    raise EmbedError(f"文本编码失败: {exc}") from exc
        if self._external is not None:
            return self._external.embed(cleaned)
        raise EmbedError("没有可用的文本编码器")
