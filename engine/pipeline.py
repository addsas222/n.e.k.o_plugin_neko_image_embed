"""四阶段流水线编排：闸门 → 判定 → 压缩 → 嵌入入库。

错误语义刻意分成两类，避免「拒收」和「坏了」混在一起：

* **拒收**（``accepted=False``）是正常结果：图太小、糊、判官说不要、风险过高。
  调用方拿到 200 语义的结果，里面写明原因。
* **故障**（``PipelineError``）是基础设施问题：编码器不可用、数据库写不进、
  压缩预算压不下去。这类必须让入口返回 ``Err``，不能伪装成「不收录」。
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .compress import CompressError, compress, make_thumb
from .config import Settings
from .embed.registry import EmbeddingRegistry, EmbedError
from .guard import digest_for_prompt, inspect_image
from .judge import JudgeChain, JudgeChainError
from .judges.base import JudgeRequest, Verdict
from .lexicon import translate_query
from .media import atomic_write, safe_relative, sha256_bytes
from .store import ImageRow, ImageStore, StoreError

_PREVIEW_EDGE = 512
_PREVIEW_QUALITY = 80
_PREVIEW_MAX_BYTES = 200 * 1024


class PipelineError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class StageResult:
    name: str
    ms: float
    ok: bool
    detail: str = ""


@dataclass
class IngestResult:
    accepted: bool
    code: str
    detail: str = ""
    image_id: int = 0
    duplicate: bool = False
    verdict: Verdict | None = None
    errors: dict[str, str] = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)
    media: dict[str, object] = field(default_factory=dict)
    row: ImageRow | None = None
    embed_mode: str = ""

    def as_dict(self) -> dict:
        payload: dict[str, object] = {
            "accepted": self.accepted,
            "code": self.code,
            "duplicate": self.duplicate,
            "id": self.image_id,
            "stages": [
                {"stage": s.name, "ms": round(s.ms, 1), "ok": s.ok, "detail": s.detail}
                for s in self.stages
            ],
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.verdict is not None:
            payload["verdict"] = {
                "engine": self.verdict.engine,
                "perception": self.verdict.perception,
                "modality": self.verdict.modality,
                "should_embed": self.verdict.should_embed,
                "value": round(self.verdict.value, 4),
                "risk_max": round(self.verdict.risk_max, 4),
                "sticker": self.verdict.sticker,
                "caption": self.verdict.caption,
                "tags": self.verdict.tags,
                "reason": self.verdict.reason,
                "issues": list(self.verdict.issues),
            }
        if self.errors:
            payload["errors"] = dict(self.errors)
        if self.media:
            payload["media"] = dict(self.media)
        if self.embed_mode:
            payload["embed_mode"] = self.embed_mode
        return payload


class IngestPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        archives: object,
        judge: JudgeChain,
        embedder: EmbeddingRegistry,
    ) -> None:
        self._settings = settings
        self._archives = archives
        self._judge = judge
        self._embedder = embedder

    # ------------------------------------------------------------------ 收录

    async def run(
        self,
        data: bytes,
        *,
        mime: str = "",
        lanlan: str = "",
        persona_text: str = "",
        force: bool = False,
        dry_run: bool = False,
    ) -> IngestResult:
        stages: list[StageResult] = []
        limits = self._settings.limits
        storage = self._settings.storage
        if not (lanlan or "").strip():
            raise PipelineError("no_persona", "缺少角色，拒绝入库")
        store = self._archives.store_for(lanlan)
        media_root = self._archives.media_root(lanlan)

        started = time.perf_counter()
        guard = inspect_image(data, limits, allow_animated=storage.allow_animated)
        stages.append(
            StageResult("guard", (time.perf_counter() - started) * 1000, guard.ok, guard.code)
        )
        if not guard.ok:
            return IngestResult(
                accepted=False, code=guard.code, detail=guard.detail, stages=stages
            )
        assert guard.image is not None and guard.metrics is not None

        sha = sha256_bytes(data)
        existing = await asyncio.to_thread(store.find_by_sha, sha, lanlan)
        if existing is not None and not force:
            return IngestResult(
                accepted=True,
                code="duplicate_exact",
                detail="同一张图已收录",
                image_id=existing.id,
                duplicate=True,
                stages=stages,
                row=existing,
            )

        near = await asyncio.to_thread(
            store.find_near_duplicate,
            guard.metrics.dhash,
            lanlan,
            max_hamming=limits.dup_hamming_max,
        )
        if near is not None and not force:
            return IngestResult(
                accepted=True,
                code="duplicate_near",
                detail=f"与 #{near.id} 视觉近似（dHash 距 ≤ {limits.dup_hamming_max}）",
                image_id=near.id,
                duplicate=True,
                stages=stages,
                row=near,
            )

        preview_started = time.perf_counter()
        preview_b64 = _preview_b64(guard.image)
        stages.append(
            StageResult(
                "preview",
                (time.perf_counter() - preview_started) * 1000,
                bool(preview_b64),
                f"{len(preview_b64)} b64",
            )
        )

        judge_started = time.perf_counter()
        request = JudgeRequest(
            persona=persona_text,
            digest=digest_for_prompt(guard.metrics, guard.probe),
            image_b64=preview_b64,
            mime="image/jpeg",
            metrics=guard.metrics,
            probe=guard.probe,
        )
        try:
            outcome = await self._judge.judge(request)
        except JudgeChainError as exc:
            raise PipelineError("judge_unavailable", str(exc.errors)) from exc
        verdict = outcome.verdict
        stages.append(
            StageResult(
                "judge",
                (time.perf_counter() - judge_started) * 1000,
                True,
                f"{verdict.perception or '-'}→{verdict.engine}",
            )
        )

        if verdict.risk_max > limits.max_risk:
            return IngestResult(
                accepted=False,
                code="risk_too_high",
                detail=f"风险 {verdict.risk_max:.2f} > {limits.max_risk}",
                verdict=verdict,
                errors=outcome.errors,
                stages=stages,
            )
        if not verdict.should_embed:
            return IngestResult(
                accepted=False,
                code="judge_declined",
                detail=verdict.reason or "判定为不值得收录",
                verdict=verdict,
                errors=outcome.errors,
                stages=stages,
            )
        if verdict.value < limits.min_value and not force:
            return IngestResult(
                accepted=False,
                code="low_value",
                detail=f"价值 {verdict.value:.2f} < {limits.min_value}",
                verdict=verdict,
                errors=outcome.errors,
                stages=stages,
            )

        compress_started = time.perf_counter()
        try:
            compressed = compress(
                guard.image,
                long_edge=storage.long_edge,
                quality=storage.jpeg_quality,
                max_bytes=storage.max_bytes,
            )
        except CompressError as exc:
            raise PipelineError("compress_failed", str(exc)) from exc
        try:
            thumb = make_thumb(
                guard.image, edge=storage.thumb_edge, quality=storage.thumb_quality
            )
        except Exception:  # noqa: BLE001 - 缩略图是附属品，失败不影响主流程
            thumb = b""
        stages.append(
            StageResult(
                "compress",
                (time.perf_counter() - compress_started) * 1000,
                True,
                f"{compressed.width}x{compressed.height} {compressed.bytes}B",
            )
        )

        embed_started = time.perf_counter()
        try:
            vector, mode = await self._embedder.embed_image(
                compressed.data, caption=verdict.caption
            )
        except EmbedError as exc:
            raise PipelineError("embed_failed", str(exc)) from exc
        stages.append(
            StageResult("embed", (time.perf_counter() - embed_started) * 1000, True, mode)
        )

        media = {
            "path": f"media/{sha}.jpg",
            "thumb": f"thumbs/{sha}.jpg" if thumb else "",
            "bytes": compressed.bytes,
            "width": compressed.width,
            "height": compressed.height,
            "quality": compressed.quality,
            "flattened": guard.flattened,
        }

        if dry_run:
            return IngestResult(
                accepted=True,
                code="dry_run",
                detail="仅判定，未落盘",
                verdict=verdict,
                errors=outcome.errors,
                stages=stages,
                media=media,
                embed_mode=mode,
            )

        media_path = safe_relative(media_root, f"media/{sha}.jpg")
        thumb_path = safe_relative(media_root, f"thumbs/{sha}.jpg") if thumb else None
        try:
            await asyncio.to_thread(_write_media, media_path, compressed.data, thumb_path, thumb)
        except OSError as exc:
            raise PipelineError("write_failed", f"{type(exc).__name__}: {exc}") from exc

        try:
            image_id, created = await asyncio.to_thread(
                store.add,
                sha256=sha,
                pixel_sha=guard.metrics.pixel_sha,
                lanlan=lanlan,
                rel_path=f"media/{sha}.jpg",
                thumb_path=f"thumbs/{sha}.jpg" if thumb else "",
                width=compressed.width,
                height=compressed.height,
                size_bytes=compressed.bytes,
                mime=compressed.mime,
                caption=verdict.caption,
                tags=verdict.tags,
                sticker=verdict.sticker,
                verdict="accepted",
                engine=verdict.engine,
                modality=verdict.modality,
                value=verdict.value,
                risk_max=verdict.risk_max,
                reason=verdict.reason,
                dhash=guard.metrics.dhash,
                vector=vector,
            )
        except StoreError as exc:
            _remove_quietly(media_path)
            if thumb_path is not None:
                _remove_quietly(thumb_path)
            raise PipelineError("store_failed", str(exc)) from exc

        row = await asyncio.to_thread(store.get, image_id)
        return IngestResult(
            accepted=created,
            code="created" if created else "duplicate_exact",
            image_id=image_id,
            duplicate=not created,
            verdict=verdict,
            errors=outcome.errors,
            stages=stages,
            media=media,
            row=row,
            embed_mode=mode,
        )

    # ------------------------------------------------------------------ 检索

    async def embed_image_bytes(self, data: bytes, *, caption: str = "") -> np.ndarray:
        """把任意图片字节编码成查询向量（以图搜图 / 按 id 搜相似）。"""
        try:
            vector, _mode = await self._embedder.embed_image(data, caption=caption)
        except EmbedError as exc:
            raise PipelineError("embed_failed", str(exc)) from exc
        return vector

    async def search(
        self, query: str, *, lanlan: str, top_k: int
    ) -> tuple[list[tuple[ImageRow, float]], bool]:
        if not (lanlan or "").strip():
            raise PipelineError("no_persona", "缺少角色，无法检索")
        store = self._archives.store_for(lanlan)
        # 中文查询先过别名表再交给 CLIP 文本塔；词法加权仍用原查询。
        translated = translate_query(query, self._settings.query_aliases) or query
        vector = await self._embedder.embed_text(translated)
        return store.search(
            vector,
            lanlan=lanlan,
            top_k=top_k,
            max_scan_rows=self._settings.storage.max_scan_rows,
            query_text=query,
        )


def _preview_b64(image: Image.Image) -> str:
    """判定用的预览图：512px、q80，控制在 200KB 内，避免把网关 payload 顶爆。"""
    preview = image.convert("RGB")
    preview.thumbnail((_PREVIEW_EDGE, _PREVIEW_EDGE), Image.Resampling.LANCZOS)
    for quality in (80, 72, 64, 56):
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=quality, optimize=True)
        payload = buffer.getvalue()
        if len(payload) <= _PREVIEW_MAX_BYTES:
            return base64.b64encode(payload).decode("ascii")
    return base64.b64encode(payload).decode("ascii")


def _write_media(media_path: Path, payload: bytes, thumb_path: Path | None, thumb: bytes) -> None:
    atomic_write(media_path, payload)
    if thumb and thumb_path is not None:
        atomic_write(thumb_path, thumb)


def _remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def collect_garbage(
    store: ImageStore,
    media_root: Path,
    *,
    deleted_before: float,
    limit: int,
    keep: int | None = None,
) -> dict[str, object]:
    """硬删墓碑与超额旧记录。路径一律走 safe_relative，绝不越出 media_root。"""
    removed: list[int] = []
    failures: list[str] = []

    targets = store.take_deleted(before=deleted_before, limit=limit)
    if keep is not None and len(targets) < limit:
        targets += store.take_oldest(keep=keep, limit=limit - len(targets))

    seen: set[int] = set()
    for row in targets:
        if row.id in seen:
            continue
        seen.add(row.id)
        for rel in (row.rel_path, row.thumb_path):
            if not rel:
                continue
            try:
                target = safe_relative(media_root, rel)
            except ValueError as exc:
                failures.append(f"#{row.id}: {exc}")
                continue
            _remove_quietly(target)
        try:
            store.hard_delete(row.id)
            removed.append(row.id)
        except Exception as exc:  # noqa: BLE001 - 单条失败不阻断整批
            failures.append(f"#{row.id}: {type(exc).__name__}: {exc}")

    return {"removed": removed, "failures": failures}
