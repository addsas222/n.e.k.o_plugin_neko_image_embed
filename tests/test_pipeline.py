"""流水线测试：拒收与故障分离、去重、dry_run、GC 不越界。"""

from __future__ import annotations

import asyncio
import hashlib
import io

import numpy as np
import pytest
from neko_image_embed.engine.config import load_settings
from neko_image_embed.engine.judge import JudgeChain
from neko_image_embed.engine.media import safe_relative
from neko_image_embed.engine.personas import PersonaArchives
from neko_image_embed.engine.pipeline import IngestPipeline, PipelineError, collect_garbage
from PIL import Image

DIM = 32


class FakeEmbedder:
    """确定性的假编码器：不加载 ONNX，只验证编排。"""

    async def embed_image(self, jpeg: bytes, *, caption: str = ""):
        digest = hashlib.sha256(jpeg).digest()
        vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        return vector / np.linalg.norm(vector), "fake"

    async def embed_text(self, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        return vector / np.linalg.norm(vector)


def _jpeg(seed: int = 5, size: tuple[int, int] = (640, 480)) -> bytes:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, "RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _build(tmp_path, raw_settings: dict | None = None):
    settings = load_settings(raw_settings or {})
    archives = PersonaArchives(tmp_path / "images", embedder_id="fake-embedder", dim=DIM)
    store = archives.store_for("YUI")
    pipeline = IngestPipeline(
        settings=settings,
        archives=archives,
        judge=JudgeChain(settings, ctx=None),
        embedder=FakeEmbedder(),
    )
    return settings, store, pipeline


def test_accepts_and_persists(tmp_path) -> None:
    settings, store, pipeline = _build(tmp_path)
    try:
        data = _jpeg()
        result = asyncio.run(
            pipeline.run(data, mime="image/jpeg", lanlan="YUI", persona_text="角色：YUI")
        )
        assert result.accepted is True
        assert result.code == "created"
        assert result.verdict is not None and result.verdict.engine == "rules"
        assert (tmp_path / "images" / "YUI" / "media" / f"{hashlib.sha256(data).hexdigest()}.jpg").is_file()
        assert (tmp_path / "images" / "YUI" / "thumbs" / f"{hashlib.sha256(data).hexdigest()}.jpg").is_file()
        assert store.stats()["live"] == 1
    finally:
        store.close()


def test_second_ingest_is_deduplicated(tmp_path) -> None:
    _settings, store, pipeline = _build(tmp_path)
    try:
        data = _jpeg()
        first = asyncio.run(pipeline.run(data, lanlan="YUI"))
        second = asyncio.run(pipeline.run(data, lanlan="YUI"))
        assert first.image_id == second.image_id
        assert second.duplicate is True
        assert store.stats()["total"] == 1
    finally:
        store.close()


def test_dry_run_writes_nothing(tmp_path) -> None:
    _settings, store, pipeline = _build(tmp_path)
    try:
        result = asyncio.run(pipeline.run(_jpeg(seed=9), lanlan="YUI", dry_run=True))
        assert result.accepted is True and result.code == "dry_run"
        assert not (tmp_path / "images" / "YUI" / "media").exists() or not any(
            (tmp_path / "images" / "YUI" / "media").iterdir()
        )
        assert store.stats()["total"] == 0
    finally:
        store.close()


def test_low_value_is_rejected_not_error(tmp_path) -> None:
    """模型说「可以收」但价值分不够时，走 low_value 而不是硬收。"""

    class StubJudge:
        async def judge(self, request):
            from neko_image_embed.engine.judge import JudgeOutcome
            from neko_image_embed.engine.judges.base import Verdict

            verdict = Verdict(
                should_embed=True,
                value=0.10,
                caption="随便一张",
                engine="stub",
                modality="vision",
            )
            return JudgeOutcome(verdict=verdict)

    settings = load_settings({"image_embed": {"min_value": 0.45}})
    archives = PersonaArchives(tmp_path / "images", embedder_id="fake-embedder", dim=DIM)
    store = archives.store_for("YUI")
    pipeline = IngestPipeline(
        settings=settings,
        archives=archives,
        judge=StubJudge(),
        embedder=FakeEmbedder(),
    )
    try:
        result = asyncio.run(pipeline.run(_jpeg(seed=13), lanlan="YUI"))
        assert result.accepted is False
        assert result.code == "low_value"
        assert store.stats()["total"] == 0
    finally:
        store.close()


def test_rules_below_threshold_declines(tmp_path) -> None:
    _settings, store, pipeline = _build(
        tmp_path, {"image_embed": {"min_value": 0.99, "judge": {"order": ["rules"]}}}
    )
    try:
        result = asyncio.run(pipeline.run(_jpeg(seed=17), lanlan="YUI"))
        assert result.accepted is False
        assert result.code == "judge_declined"
    finally:
        store.close()


def test_guard_rejection_is_a_result(tmp_path) -> None:
    _settings, store, pipeline = _build(tmp_path)
    try:
        flat = io.BytesIO()
        Image.new("RGB", (320, 240), (7, 7, 7)).save(flat, format="JPEG")
        result = asyncio.run(pipeline.run(flat.getvalue(), lanlan="YUI"))
        assert result.accepted is False
        assert result.code == "low_entropy"
    finally:
        store.close()


def test_embed_failure_is_an_error(tmp_path) -> None:
    settings = load_settings({"image_embed": {"judge": {"order": ["rules"]}}})

    class Broken(FakeEmbedder):
        async def embed_image(self, jpeg: bytes, *, caption: str = ""):
            from neko_image_embed.engine.embed.registry import EmbedError

            raise EmbedError("no backend")

    archives = PersonaArchives(tmp_path / "images", embedder_id="fake-embedder", dim=DIM)
    store = archives.store_for("YUI")
    pipeline = IngestPipeline(
        settings=settings,
        archives=archives,
        judge=JudgeChain(settings, ctx=None),
        embedder=Broken(),
    )
    try:
        with pytest.raises(PipelineError) as info:
            asyncio.run(pipeline.run(_jpeg(seed=21), lanlan="YUI"))
        assert info.value.code == "embed_failed"
        assert store.stats()["total"] == 0
        assert not (tmp_path / "images" / "YUI" / "media").exists()
    finally:
        store.close()


def test_gc_removes_files_and_rows(tmp_path) -> None:
    _settings, store, pipeline = _build(tmp_path)
    data = _jpeg(seed=31)
    try:
        result = asyncio.run(pipeline.run(data, lanlan="YUI"))
        assert result.accepted
        store.soft_delete(result.image_id)
        report = collect_garbage(
            store, tmp_path / "images" / "YUI", deleted_before=9e18, limit=10
        )
        assert report["failures"] == []
        assert report["removed"] == [result.image_id]
        assert store.stats()["total"] == 0
        assert not (
            tmp_path / "images" / "YUI" / "media" / f"{hashlib.sha256(data).hexdigest()}.jpg"
        ).exists()
    finally:
        store.close()


def test_safe_relative_rejects_escape(tmp_path) -> None:
    with pytest.raises(ValueError):
        safe_relative(tmp_path, "../evil.jpg")
    with pytest.raises(ValueError):
        safe_relative(tmp_path, "C:/Windows/system32/evil.jpg")
    assert safe_relative(tmp_path, "media/a.jpg") == (tmp_path / "media" / "a.jpg").resolve()


def test_search_translates_chinese_query(tmp_path) -> None:
    """中文查询要先过别名表，再交给 CLIP 文本塔。"""

    class Recording(FakeEmbedder):
        def __init__(self) -> None:
            self.texts: list[str] = []

        async def embed_text(self, text: str):
            self.texts.append(text)
            return await super().embed_text(text)

    settings = load_settings({})
    archives = PersonaArchives(tmp_path / "images", embedder_id="fake-embedder", dim=DIM)
    store = archives.store_for("YUI")
    embedder = Recording()
    pipeline = IngestPipeline(
        settings=settings,
        archives=archives,
        judge=JudgeChain(settings, ctx=None),
        embedder=embedder,
    )
    try:
        asyncio.run(pipeline.search("猫", lanlan="YUI", top_k=3))
        assert embedder.texts == ["cat"]
    finally:
        store.close()
