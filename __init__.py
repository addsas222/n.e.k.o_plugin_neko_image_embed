"""N.E.K.O 图像记忆库插件。

流水线：合规/质量闸门 → 判定链（gateway → laya → kev → semif → jev → rules）
→ 压缩 → CLIP 向量入库；聊天时经 LLM 工具检索并发图。

三个必须写在代码里的 SDK 约束（都在 SDK 源码里核实过）：

* 入口一律 ``async def``。宿主对同步入口直接报
  ``Entry must be 'async def'`` 并拒绝执行。
* 入口必须吃掉 ``**_``：宿主会额外注入 ``_ctx``，签名写死会被打爆。
* ``ctx.images.upload`` 在 lifecycle 里调用必抛 ``RuntimeError``，所以发图只能在
  入口/LLM 工具里做；lifecycle 只做目录与状态。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import inspect
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from plugin.sdk.plugin import (
    Err,
    NekoPluginBase,
    Ok,
    SdkError,
    lifecycle,
    llm_tool,
    neko_plugin,
    plugin_entry,
    timer_interval,
)

from .engine.config import Settings, load_settings
from .engine.embed.registry import EmbeddingRegistry
from .engine.judge import JudgeChain
from .engine.judges.gateway import ensure_sdk_imported
from .engine.media import human_bytes
from .engine.persona import default_characters_path, load_persona
from .engine.personas import PersonaArchives
from .engine.pipeline import IngestPipeline, PipelineError, collect_garbage
from .engine.store import EmbedderMismatch, StoreError

_INGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {
            "type": "string",
            "enum": ["frames", "path", "url", "b64"],
            "default": "frames",
            "description": "图片来源：最近一次共享画面 / 本机路径 / HTTP(S) 地址 / base64",
        },
        "path": {"type": "string", "description": "source=path 时的本机文件路径"},
        "url": {"type": "string", "description": "source=url 时的 HTTP(S) 地址"},
        "b64": {"type": "string", "description": "source=b64 时的 base64（可带 data: 前缀）"},
        "mime": {"type": "string", "description": "仅作提示，实际以解码结果为准"},
        "lanlan": {"type": "string", "description": "归属角色，缺省取当前会话目标"},
        "force": {"type": "boolean", "default": False, "description": "跳过去重与价值阈值"},
        "dry_run": {"type": "boolean", "default": False, "description": "只判定不落盘"},
    },
}

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "文字查询；与 image_b64 至少给一个"},
        "image_b64": {"type": "string", "description": "以图搜图"},
        "image_id": {"type": "integer", "description": "以已收录的某张图搜相似图"},
        "top_k": {"type": "integer", "default": 8},
        "lanlan": {"type": "string", "description": "只搜该角色的记忆；缺省用当前角色"},
    },
}

_SEND_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {"type": "integer", "description": "要发送的图片 id"},
        "lanlan": {"type": "string", "description": "目标角色"},
    },
    "required": ["image_id"],
}

_FORGET_SCHEMA = {
    "type": "object",
    "properties": {"image_id": {"type": "integer"}},
    "required": ["image_id"],
}

_GC_SCHEMA = {
    "type": "object",
    "properties": {
        "persona": {"type": "string", "description": "目标角色存档；缺省为当前角色"},
        "lanlan": {"type": "string", "description": "同 persona（兼容旧调用）"},
        "days": {"type": "integer", "default": 30, "description": "墓碑保留天数"},
        "limit": {"type": "integer", "default": 200, "description": "本次最多处理多少条"},
        "keep": {"type": "integer", "default": 0, "description": ">0 时把存活记录裁剪到该数量"},
    },
}

_BOARD_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "留空则列出最近收录"},
        "limit": {"type": "integer", "default": 12, "description": "返回条数，最多 24"},
        "include_thumb": {"type": "boolean", "default": True, "description": "附带缩略图 data URL"},
        "persona": {"type": "string", "description": "查看哪一档；缺省用当前角色"},
        "lanlan": {"type": "string", "description": "同 persona（兼容旧调用）"},
    },
}

_REASSIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {"type": "integer", "description": "要迁移的图片 id（在源角色存档里）"},
        "lanlan": {"type": "string", "description": "目标角色（必填）"},
        "from_lanlan": {"type": "string", "description": "源角色，缺省为当前角色"},
    },
    "required": ["image_id", "lanlan"],
}

_JUDGE_MODE_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["model", "host", "heuristic"],
            "description": "判断层：模型 / 宿主槽位 / 启发式",
        }
    },
    "required": ["mode"],
}

_LOG_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "default": 40, "description": "返回条数，最多 200"},
        "level": {
            "type": "string",
            "enum": ["", "info", "warn", "error"],
            "description": "按级别过滤；留空为全部",
        },
        "persona": {"type": "string", "description": "查看哪一档的日志；缺省用当前角色"},
    },
}


def _extract_items(result: object) -> list[Any]:
    """bus 的返回可能是 SdkBusList / Ok / list / 可迭代，统一摊平成 list。"""
    if result is None:
        return []
    items = getattr(result, "items", None)
    if isinstance(items, list):
        return items
    value = getattr(result, "value", None)
    if isinstance(value, list):
        return value
    if isinstance(result, list):
        return result
    try:
        return list(result)  # type: ignore[arg-type]
    except TypeError:
        return []


def _receipt_state(receipt: object) -> tuple[bool, str]:
    if not isinstance(receipt, dict):
        return True, ""
    if receipt.get("submitted"):
        return True, ""
    return False, str(receipt.get("reason") or "rejected")


def _strip_data_url(value: str) -> tuple[str, str]:
    text = (value or "").strip()
    if text.startswith("data:"):
        header, _, payload = text.partition(",")
        mime = header[5:].split(";")[0] or "image/jpeg"
        return payload, mime
    return text, ""


@neko_plugin
class NekoImageEmbedPlugin(NekoPluginBase):
    """图像记忆库：收录、检索、发图。"""

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        # 用 threading.Lock 而不是 asyncio.Lock：宿主把 entry、timer、startup 放在
        # 不同的事件循环里跑，asyncio 原语跨 loop 会 "bound to a different loop"。
        self._init_guard = threading.Lock()
        self._settings: Settings | None = None
        self._archives: PersonaArchives | None = None
        self._embedder: EmbeddingRegistry | None = None
        self._judge: JudgeChain | None = None
        self._pipeline: IngestPipeline | None = None
        self._ready = False
        self._init_error = ""
        self._init_failed_at = 0.0
        self._persona_cache = None
        self._persona_key: tuple[str, str] = ("", "")
        self._persona_at = 0.0
        self._last_frame: str = ""
        self._warm_imports = False

    # ------------------------------------------------------------------ 基础设施

    def _host_root(self) -> Path:
        try:
            return self.storage_dir.parents[1]
        except (AttributeError, IndexError):
            return Path.cwd()

    def _log(self, level: str, message: str) -> None:
        logger = getattr(self, "logger", None) or getattr(getattr(self, "ctx", None), "logger", None)
        if logger is None:
            return
        emit = getattr(logger, level, None) or getattr(logger, "info", None)
        if emit is None:
            return
        with contextlib.suppress(Exception):  # 日志永远不能影响业务
            emit(message)

    async def _event(
        self,
        kind: str,
        message: str,
        *,
        level: str = "info",
        image_id: int = 0,
        engine: str = "",
        ms: float = 0.0,
        persona: str = "",
    ) -> None:
        """写一条活动记录进**该角色自己的**索引库。失败绝不影响主流程。"""
        archives = self._archives
        if archives is None:
            return
        name = persona or await self._persona_identity()
        if not name:
            return
        with contextlib.suppress(Exception):
            store = await asyncio.to_thread(archives.store_for, name)
            await asyncio.to_thread(
                store.log_event,
                kind,
                message,
                level=level,
                image_id=image_id,
                engine=engine,
                ms=ms,
            )

    def _judge_mode_path(self) -> Path:
        return self.data_path("judge_mode.json")

    def _apply_judge_override(self, settings: Settings) -> Settings:
        """面板切换判断层写的是插件自己的小文件，绕开宿主 config.set 的超时。"""
        path = self._judge_mode_path()
        if not path.is_file():
            return settings
        try:
            mode = str(json.loads(path.read_text(encoding="utf-8")).get("mode") or "")
        except (OSError, ValueError):
            return settings
        if mode not in {"model", "host", "heuristic"}:
            return settings
        return replace(settings, judge=replace(settings.judge, mode=mode))

    async def _settings_snapshot(self) -> Settings:
        if self._settings is not None:
            return self._settings
        raw: object = {}
        try:
            dumped = await self.config.dump()
            raw = dumped
        except Exception as exc:  # noqa: BLE001 - 配置后端异常时退默认值
            self._log("warning", f"读取配置失败，使用默认值：{type(exc).__name__}: {exc}")
        settings = self._apply_judge_override(load_settings(raw))
        self._settings = settings
        return settings

    def _fail_init(self, detail: str, level: str = "warning") -> tuple[bool, str]:
        """记录一次初始化失败，并允许 60 秒后重试（下载/服务可能只是暂时不可用）。"""
        self._init_error = detail
        self._init_failed_at = time.time()
        self._log(level, detail)
        return False, detail

    def _model_present(self, settings: Settings) -> bool:
        """模型是否已落在本地。只查文件存在，不校验哈希——用来决定 startup 能否同步初始化。"""
        if settings.embedder.backend == "external":
            return True
        try:
            registry = EmbeddingRegistry(
                settings.embedder,
                data_dir=self.data_path(),
                plugin_dir=self.plugin_dir,
                timeout=5.0,
            )
            model_dir = registry.model_dir()
        except Exception:  # noqa: BLE001 - 路径算不出来就当作没有
            return False
        return all((model_dir / name).is_file() for name in ("model.onnx", "tokenizer.json"))

    async def _ensure(self) -> tuple[bool, str]:
        """惰性初始化：编码器就绪 → 开库 → 装配流水线。

        会在三个不同的事件循环里被调用（entry / timer / startup），所以：
        * 互斥用 ``threading.Lock``，不用 ``asyncio.Lock``（跨 loop 会报
          "bound to a different loop"）；拿不到锁就轮询等它就绪，不阻塞事件循环；
        * 失败有 60 秒冷却，避免每个 timer tick 都重试一次下载。
        """
        if self._ready:
            return True, ""
        if self._init_error and time.time() - self._init_failed_at < 60:
            return False, self._init_error

        if not self._init_guard.acquire(blocking=False):
            for _ in range(100):
                await asyncio.sleep(0.1)
                if self._ready:
                    return True, ""
                if self._init_error:
                    return False, self._init_error
            return False, "初始化正在进行中"

        try:
            if self._ready:
                return True, ""
            if self._init_error and time.time() - self._init_failed_at < 60:
                return False, self._init_error
            self._init_error = ""
            try:
                settings = await self._settings_snapshot()
                if "zeroshot" not in settings.judge.perception:
                    self._log(
                        "warning",
                        "识图链（judge.perception）里没有 zeroshot：Sensenova 不可用时会没有 "
                        "caption/tags 兜底；建议保留 perception = [\"sensenova\", \"zeroshot\"]",
                    )
                embedder = EmbeddingRegistry(
                    settings.embedder,
                    data_dir=self.data_path(),
                    plugin_dir=self.plugin_dir,
                    timeout=120.0,
                )
                state = await embedder.prepare()
                if not state.ready:
                    return self._fail_init(f"编码器不可用：{state.detail}")

                # 每个角色一套独立存档：data/images/<角色>/{index.db,media,thumbs}
                archives = PersonaArchives(
                    self.data_path("images"),
                    embedder_id=state.id,
                    dim=state.dim,
                )
                current = await self._persona_identity()
                if current:
                    try:
                        await asyncio.to_thread(archives.store_for, current)
                        migrated = await asyncio.to_thread(
                            archives.migrate_legacy, default_persona=current
                        )
                        if migrated.get("migrated"):
                            self._log(
                                "info",
                                f"旧版单库已按角色拆分：迁移 {migrated['migrated']} 条"
                                + (f"，失败 {len(migrated['failures'])} 条" if migrated["failures"] else ""),
                            )
                    except (EmbedderMismatch, StoreError, ValueError) as exc:
                        await asyncio.to_thread(archives.close)
                        return self._fail_init(f"存档打开失败：{type(exc).__name__}: {exc}", level="error")

                judge = JudgeChain(
                    settings,
                    ctx=self.ctx,
                    embedder=embedder,
                    cache_dir=self.cache_path("zeroshot"),
                )
                self._archives = archives
                self._embedder = embedder
                self._judge = judge
                self._pipeline = IngestPipeline(
                    settings=settings,
                    archives=archives,
                    judge=judge,
                    embedder=embedder,
                )
                self._ready = True
                self._init_error = ""
                self._init_failed_at = 0.0
                return True, ""
            except Exception as exc:  # noqa: BLE001 - 初始化阶段的任何意外都要收敛
                return self._fail_init(f"初始化失败：{type(exc).__name__}: {exc}", level="error")
        finally:
            self._init_guard.release()

    async def _require(self) -> tuple[IngestPipeline | None, PersonaArchives | None]:
        ok, detail = await self._ensure()
        if not ok:
            return None, None
        return self._pipeline, self._archives

    async def _persona_text(self) -> str:
        settings = await self._settings_snapshot()
        path = settings.characters_path or str(default_characters_path(self._host_root()))
        key = (path, settings.persona_summary)
        now = time.time()
        if self._persona_cache is not None and self._persona_key == key and now - self._persona_at < 60:
            return self._persona_cache.text
        persona = await asyncio.to_thread(
            load_persona, override=settings.persona_summary, characters_path=path
        )
        if persona.warning:
            self._log("info", f"人设回退：{persona.warning}")
        self._persona_cache = persona
        self._persona_key = key
        self._persona_at = now
        return persona.text

    @staticmethod
    def _resolve_lanlan(explicit: str, kwargs: dict) -> str:
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        inner = kwargs.get("_ctx")
        if isinstance(inner, dict):
            name = inner.get("lanlan_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return ""

    def _current_lanlan(self) -> str:
        value = getattr(self.ctx, "current_lanlan", None)
        return value.strip() if isinstance(value, str) else ""

    async def _persona_identity(self, explicit: str = "", kwargs: dict | None = None) -> str:
        """当前角色是谁：显式参数 → 宿主上下文 → characters.json 的「当前猫娘」。

        面板与外部调用常常拿不到 ``ctx.current_lanlan``，这里必须退到宿主自己的
        人设配置，否则会静默落进「公共桶」（lanlan=''），人设隔离就形同虚设。
        """
        name = self._resolve_lanlan(explicit, kwargs or {}) or self._current_lanlan()
        if name:
            return name
        await self._persona_text()
        persona = self._persona_cache
        if persona is not None and persona.name:
            return str(persona.name)
        return ""

    # ------------------------------------------------------------------ 图片来源

    def _media_file(self, rel_path: str, persona: str = "") -> Path:
        from .engine.media import safe_relative

        if persona and self._archives is not None:
            base = self._archives.media_root(persona)
        else:
            base = self.data_path("images")
        return safe_relative(base, rel_path)

    async def _store_for(self, persona: str) -> Any:
        archives = self._archives
        if archives is None or not (persona or "").strip():
            raise StoreError("缺少角色或插件尚未就绪")
        return await asyncio.to_thread(archives.store_for, persona)

    async def _load_source(
        self, source: str, *, path: str, url: str, b64: str, mime: str
    ) -> tuple[bytes, str]:
        kind = (source or "frames").strip().lower()
        settings = await self._settings_snapshot()
        limit = settings.limits.max_source_bytes

        if kind == "frames":
            return await self._frame_bytes()
        if kind == "path":
            target = Path(path).expanduser()
            if not target.is_file():
                raise ValueError(f"文件不存在：{target}")
            if target.stat().st_size > limit:
                raise ValueError(f"文件 {human_bytes(target.stat().st_size)} 超过上限 {human_bytes(limit)}")
            return await asyncio.to_thread(target.read_bytes), mime or ""
        if kind == "url":
            return await asyncio.to_thread(self._fetch_url, url, limit), mime or ""
        if kind == "b64":
            payload, sniffed = _strip_data_url(b64)
            try:
                data = base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"base64 解码失败：{exc}") from exc
            if not data:
                raise ValueError("base64 内容为空")
            if len(data) > limit:
                raise ValueError(f"图片 {human_bytes(len(data))} 超过上限 {human_bytes(limit)}")
            return data, mime or sniffed
        raise ValueError(f"未知 source：{source!r}")

    def _fetch_url(self, url: str, limit: int) -> bytes:
        parsed = urlsplit(url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("只接受 http/https 图片地址")
        request = urllib.request.Request(url, headers={"user-agent": "neko-image-embed/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                chunks: list[bytes] = []
                total = 0
                while True:
                    block = response.read(64 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > limit:
                        raise ValueError(f"远端图片超过上限 {human_bytes(limit)}")
                    chunks.append(block)
        except urllib.error.HTTPError as exc:
            raise ValueError(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ValueError(f"下载失败：{exc}") from exc
        if not chunks:
            raise ValueError("远端返回空内容")
        return b"".join(chunks)

    async def _frame_bytes(self) -> tuple[bytes, str]:
        bus = getattr(self.ctx, "bus", None)
        frames = getattr(bus, "frames", None)
        if frames is None:
            raise ValueError("ctx.bus.frames 不可用，无法读取共享画面")
        try:
            result = frames.get(max_count=8)
        except TypeError:
            result = frames.get()
        if inspect.isawaitable(result):
            result = await result
        items = _extract_items(result)
        candidates = [
            record
            for record in items
            if getattr(record, "image_base64", None)
            and str(getattr(record, "source", "")) in {"user", "screen", "camera"}
        ]
        if not candidates:
            raise ValueError("最近没有可用的共享画面（frames 为空或已过期）")
        record = max(candidates, key=lambda item: float(getattr(item, "captured_at", 0) or 0))
        self._last_frame = str(getattr(record, "frame_id", "") or "")
        payload, sniffed = _strip_data_url(str(record.image_base64))
        data = base64.b64decode(payload)
        return data, str(getattr(record, "mime", "") or sniffed or "image/jpeg")

    # ------------------------------------------------------------------ 生命周期

    @lifecycle(id="startup")
    async def on_startup(self, **_):
        for part in ("images", "images/media", "images/thumbs"):
            try:
                self.data_path(*part.split("/")).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self._log("warning", f"创建目录失败 {part}: {exc}")
        settings = await self._settings_snapshot()

        # 不要用 asyncio.create_task 做预热：宿主跑完 startup 就关掉那个 loop，
        # 挂起的任务会被回收。实测 _ensure 从未执行，img_status 一直停在 initializing。
        # 模型已在本地（约 2 秒）就同步初始化；否则交给 img_watch 的首次 tick 或 img_status。
        deferred = ""
        if self._model_present(settings):
            ok, detail = await self._ensure()
            if not ok:
                deferred = detail
        else:
            deferred = "模型未下载：将由 img_watch 首次 tick 或 img_status 触发初始化"
            self._log("warning", deferred)

        self.report_status({"state": "ready" if self._ready else "deferred", "detail": deferred})
        return Ok(
            {
                "status": "ready" if self._ready else "deferred",
                "deferred": deferred,
                "warnings": list(settings.warnings)[:8],
                "watch_frames": settings.watch_frames,
            }
        )

    @lifecycle(id="shutdown")
    async def on_shutdown(self, **_):
        if self._archives is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._archives.close)
        self._ready = False
        return Ok({"status": "stopped"})

    @lifecycle(id="config_change")
    async def on_config_change(self, **_):
        if self._archives is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._archives.close)
        self._settings = None
        self._archives = None
        self._embedder = None
        self._judge = None
        self._pipeline = None
        self._persona_cache = None
        self._persona_at = 0.0
        self._ready = False
        self._init_error = ""
        return Ok({"status": "config_reloaded"})

    # ------------------------------------------------------------------ 入口

    @plugin_entry(
        id="img_ingest",
        name="收录图片",
        description="对一张图跑完整流水线：合规闸门 → 人设判定 → 压缩 → 向量化入库。",
        input_schema=_INGEST_SCHEMA,
        timeout=180.0,
        llm_result_fields=["accepted", "code", "id", "detail", "verdict"],
    )
    async def img_ingest(
        self,
        source: str = "frames",
        path: str = "",
        url: str = "",
        b64: str = "",
        mime: str = "",
        lanlan: str = "",
        force: bool = False,
        dry_run: bool = False,
        **kwargs,
    ):
        pipeline, _store = await self._require()
        if pipeline is None:
            return Err(SdkError(self._init_error or "插件尚未就绪", code="NOT_READY"))
        target = await self._persona_identity(lanlan, kwargs)
        if not target:
            await self._event("ingest", "拒绝：无法确定当前角色", level="warn")
            return Err(
                SdkError(
                    "无法确定当前角色，拒绝入库以免落进公共桶；请显式传 lanlan，"
                    "或确认宿主 config/characters.json 里有「当前猫娘」",
                    code="NO_PERSONA",
                )
            )
        try:
            data, sniffed = await self._load_source(
                source, path=path, url=url, b64=b64, mime=mime
            )
        except ValueError as exc:
            await self._event("ingest", f"拒绝：{exc}", level="warn", persona=target)
            return Err(SdkError(str(exc), code="INVALID_ARGUMENT", details={"source": source}))

        persona = await self._persona_text()
        try:
            result = await pipeline.run(
                data,
                mime=sniffed,
                lanlan=target,
                persona_text=persona,
                force=bool(force),
                dry_run=bool(dry_run),
            )
        except PipelineError as exc:
            self._log("error", f"收录失败 {exc.code}: {exc.detail}")
            await self._event("ingest", f"{exc.code}: {exc.detail}", level="error", persona=target)
            return Err(SdkError(str(exc), code=exc.code.upper()))
        except Exception as exc:  # noqa: BLE001 - 入口边界不向上冒泡
            self._log("error", f"收录异常：{type(exc).__name__}: {exc}")
            await self._event("ingest", f"异常：{type(exc).__name__}: {exc}", level="error", persona=target)
            return Err(SdkError(f"收录异常：{type(exc).__name__}: {exc}", code="INGEST_FAILED"))

        await self._event(
            "ingest",
            f"{result.code}" + (f" · {result.detail}" if result.detail else ""),
            level="info" if result.accepted else "warn",
            image_id=result.image_id,
            engine=(result.verdict.engine if result.verdict else ""),
            ms=sum(stage.ms for stage in result.stages),
            persona=target,
        )
        return Ok(result.as_dict())

    @plugin_entry(
        id="img_search",
        name="检索图片",
        description="用文字、另一张图或已有 id 检索图像记忆，返回相似度排序。",
        input_schema=_SEARCH_SCHEMA,
        timeout=60.0,
        llm_result_fields=["count", "results"],
    )
    async def img_search(
        self,
        query: str = "",
        image_b64: str = "",
        image_id: int = 0,
        top_k: int = 0,
        lanlan: str = "",
        **kwargs,
    ):
        pipeline, archives = await self._require()
        if pipeline is None or archives is None:
            return Err(SdkError(self._init_error or "插件尚未就绪", code="NOT_READY"))
        settings = await self._settings_snapshot()
        limit = max(1, min(int(top_k or settings.storage.default_top_k), settings.storage.max_search_k))

        scope = await self._persona_identity(lanlan, kwargs)
        if not scope:
            return Err(
                SdkError("无法确定当前角色；请显式传 lanlan", code="NO_PERSONA")
            )
        store = await self._store_for(scope)

        try:
            if image_b64.strip():
                payload, _ = _strip_data_url(image_b64)
                raw = base64.b64decode(payload, validate=True)
                vector = await pipeline.embed_image_bytes(raw)
                rows, truncated = store.search(
                    vector,
                    lanlan=scope,
                    top_k=limit,
                    max_scan_rows=settings.storage.max_scan_rows,
                )
            elif int(image_id or 0) > 0:
                row = store.get(int(image_id))
                if row is None:
                    return Err(SdkError(f"找不到图片 #{image_id}", code="NOT_FOUND"))
                raw = await asyncio.to_thread(self._media_file(row.rel_path, scope).read_bytes)
                vector = await pipeline.embed_image_bytes(raw, caption=row.caption)
                rows, truncated = store.search(
                    vector,
                    lanlan=scope,
                    top_k=limit,
                    max_scan_rows=settings.storage.max_scan_rows,
                )
            else:
                if not query.strip():
                    return Err(SdkError("query、image_b64、image_id 至少给一个", code="INVALID_ARGUMENT"))
                rows, truncated = await pipeline.search(query, lanlan=scope, top_k=limit)
        except (StoreError, ValueError, binascii.Error) as exc:
            return Err(SdkError(f"检索失败：{exc}", code="SEARCH_FAILED"))
        except Exception as exc:  # noqa: BLE001
            self._log("error", f"检索异常：{type(exc).__name__}: {exc}")
            return Err(SdkError(f"检索异常：{type(exc).__name__}: {exc}", code="SEARCH_FAILED"))

        return Ok(
            {
                "count": len(rows),
                "truncated": truncated,
                "results": [_row_view(row, score) for row, score in rows],
            }
        )

    @plugin_entry(
        id="img_send",
        name="发送图片",
        description="把库里的图片压缩后推进当前聊天。",
        input_schema=_SEND_SCHEMA,
        timeout=30.0,
    )
    async def img_send(self, image_id: int = 0, lanlan: str = "", **kwargs):
        _pipeline, archives = await self._require()
        if archives is None:
            return Err(SdkError(self._init_error or "插件尚未就绪", code="NOT_READY"))
        persona = await self._persona_identity(lanlan, kwargs)
        if not persona:
            return Err(SdkError("无法确定当前角色", code="NO_PERSONA"))
        store = await self._store_for(persona)
        row = await asyncio.to_thread(store.get, int(image_id or 0))
        if row is None:
            return Err(SdkError(f"找不到图片 #{image_id}", code="NOT_FOUND"))

        target = persona
        try:
            sent = await self._push_image(row, target)
        except Exception as exc:  # noqa: BLE001
            self._log("error", f"发图失败 #{row.id}: {type(exc).__name__}: {exc}")
            await self._event("send", f"#{row.id} 失败：{type(exc).__name__}: {exc}", level="error", image_id=row.id)
            return Err(SdkError(f"发图失败：{type(exc).__name__}: {exc}", code="SEND_FAILED"))
        with contextlib.suppress(StoreError):
            await asyncio.to_thread(store.touch, row.id)
        await self._event("send", f"#{row.id} → {target or '当前会话'}", image_id=row.id, persona=target)
        return Ok(sent)

    @plugin_entry(
        id="img_forget",
        name="遗忘图片",
        description="软删除一条图片记忆；文件保留，直到 img_gc 硬删。",
        input_schema=_FORGET_SCHEMA,
        timeout=15.0,
    )
    async def img_forget(self, image_id: int = 0, persona: str = "", lanlan: str = "", **kwargs):
        _pipeline, archives = await self._require()
        if archives is None:
            return Err(SdkError(self._init_error or "插件尚未就绪", code="NOT_READY"))
        # 主控操作必须能指定档位：早先只认 image_id，传入的 lanlan 被 **_ 静默吞掉，
        # 结果删的是当前角色——这条参数不是可选的礼貌，是防误删的护栏。
        persona = await self._persona_identity(persona or lanlan, kwargs)
        if not persona:
            return Err(SdkError("无法确定当前角色", code="NO_PERSONA"))
        store = await self._store_for(persona)
        try:
            removed = await asyncio.to_thread(store.soft_delete, int(image_id or 0))
        except StoreError as exc:
            return Err(SdkError(str(exc), code="STORE_ERROR"))
        if not removed:
            return Err(SdkError(f"#{image_id} 不存在或已删除", code="NOT_FOUND"))
        await self._event("forget", f"软删 #{image_id}", level="warn", image_id=int(image_id), persona=persona)
        return Ok({"forgotten": int(image_id), "mode": "soft"})

    @plugin_entry(
        id="img_gc",
        name="清理索引",
        description="清理墓碑记录与其文件，或把存活记录裁剪到指定数量。",
        input_schema=_GC_SCHEMA,
        timeout=120.0,
    )
    async def img_gc(
        self, days: int = 30, limit: int = 200, keep: int = 0, persona: str = "", lanlan: str = "", **kwargs
    ):
        _pipeline, archives = await self._require()
        if archives is None:
            return Err(SdkError(self._init_error or "插件尚未就绪", code="NOT_READY"))
        persona = await self._persona_identity(persona or lanlan, kwargs)
        if not persona:
            return Err(SdkError("无法确定当前角色", code="NO_PERSONA"))
        store = await self._store_for(persona)
        media_root = archives.media_root(persona)
        before = time.time() - max(0, int(days)) * 86400
        try:
            report = await asyncio.to_thread(
                collect_garbage,
                store,
                media_root,
                deleted_before=before,
                limit=max(1, min(int(limit), 2000)),
                keep=int(keep) if int(keep) > 0 else None,
            )
        except Exception as exc:  # noqa: BLE001
            self._log("error", f"清理失败：{type(exc).__name__}: {exc}")
            return Err(SdkError(f"清理失败：{type(exc).__name__}: {exc}", code="GC_FAILED"))
        await self._event(
            "gc",
            f"清理 {len(report['removed'])} 条"
            + (f"，失败 {len(report['failures'])} 条" if report["failures"] else ""),
            level="warn" if report["failures"] else "info",
            persona=persona,
        )
        return Ok({"removed": len(report["removed"]), "ids": report["removed"][:50],
                   "failures": report["failures"][:20]})

    @plugin_entry(
        id="img_set_judge",
        name="切换判断层",
        description="判断层三选一：模型 / 宿主槽位 / 启发式。写入运行时配置后由宿主热重载。",
        input_schema=_JUDGE_MODE_SCHEMA,
        timeout=20.0,
    )
    async def img_set_judge(self, mode: str = "", **_):
        choice = (mode or "").strip().lower()
        if choice not in {"model", "host", "heuristic"}:
            return Err(
                SdkError("mode 只能是 model / host / heuristic", code="INVALID_ARGUMENT")
            )
        from .engine.media import atomic_write

        try:
            await asyncio.to_thread(
                atomic_write,
                self._judge_mode_path(),
                json.dumps({"mode": choice}).encode("utf-8"),
            )
        except OSError as exc:
            return Err(SdkError(f"写入切换文件失败：{exc}", code="CONFIG_FAILED"))

        # 就地重建：不动宿主配置（config.set 在这台机器上会超时），也不重启进程。
        archives = self._archives
        if archives is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(archives.close)
        self._settings = None
        self._archives = None
        self._judge = None
        self._pipeline = None
        self._ready = False
        ok, detail = await self._ensure()
        if not ok:
            return Err(SdkError(f"切换后重建失败：{detail}", code="REBUILD_FAILED"))
        await self._event("judge_mode", f"判断层切换为 {choice}")
        return Ok({"mode": choice, "note": "已在插件内即时生效"})

    @plugin_entry(
        id="img_status",
        name="运行状态",
        description="编码器、判定链、存储与人设来源的自检快照。",
        timeout=30.0,
    )
    async def img_status(self, **_):
        settings = await self._settings_snapshot()
        # 模型在本地就顺手把初始化推完（约 2 秒），省得用户盯着 initializing 等 tick。
        if not self._ready and not self._init_error and self._model_present(settings):
            await self._ensure()
        if self._persona_cache is None:
            with contextlib.suppress(Exception):
                await self._persona_text()
        if self._ready:
            state = "ready"
        elif self._init_error:
            state = "degraded"
        else:
            state = "initializing"

        payload: dict[str, object] = {
            "state": state,
            "ready": self._ready,
            "init_error": self._init_error,
            "warnings": list(settings.warnings),
            "embedder": {
                "backend": self._embedder.state.backend if self._embedder else "none",
                "ready": self._embedder.state.ready if self._embedder else False,
                "detail": self._embedder.state.detail if self._embedder else "尚未初始化",
                "id": self._embedder.id if self._embedder else "",
                "dim": self._embedder.dim if self._embedder else 0,
                "model_dir": str(self._embedder.model_dir()) if self._embedder else "",
                "allow_download": settings.embedder.allow_download,
            },
            "judge": {
                "mode": settings.judge.mode,
                "perception_mode": list(settings.judge.perception),
                "active": self._judge.engine_names if self._judge else [],
                "perception": self._judge.perception_names if self._judge else [],
                "gateway_note": "宿主未绑定 [plugin.models.judge] 槽位时，首次调用会 403 并降级",
                "engines": [
                    {
                        "id": cfg.id,
                        "enabled": cfg.enabled,
                        "protocol": cfg.protocol,
                        "base_url": cfg.base_url,
                        "model": cfg.model,
                        "vision": cfg.vision,
                    }
                    for cfg in settings.judge.engines.values()
                ],
            },
            "storage": {
                "max_bytes": settings.storage.max_bytes,
                "long_edge": settings.storage.long_edge,
                "db": str(self.data_path("images")),
            },
            "persona": self._persona_status(),
            "counts": None,
            "archives": [],
        }
        if self._archives is not None:
            name = await self._persona_identity()
            if name:
                with contextlib.suppress(StoreError, OSError):
                    store = await asyncio.to_thread(self._archives.store_for, name)
                    payload["counts"] = await asyncio.to_thread(store.stats)
            with contextlib.suppress(Exception):
                archives = await asyncio.to_thread(self._archives.list_archives)
                payload["archives"] = [
                    {
                        "persona": item.persona,
                        "slug": item.slug,
                        "live": item.live,
                        "bytes": item.bytes,
                    }
                    for item in archives
                ]
        return Ok(payload)

    @plugin_entry(
        id="img_board",
        name="图片面板",
        description="给 UI 面板用：列出或搜索图片，附带缩略图 data URL。",
        input_schema=_BOARD_SCHEMA,
        timeout=60.0,
    )
    async def img_board(
        self,
        query: str = "",
        limit: int = 12,
        include_thumb: bool = True,
        lanlan: str = "",
        persona: str = "",
        **kwargs,
    ):
        pipeline, archives = await self._require()
        if pipeline is None or archives is None:
            return Err(SdkError(self._init_error or "插件尚未就绪", code="NOT_READY"))

        # 一次只打开一个角色的存档；面板用 persona 切换查看哪一档，不做跨档合并。
        target = (persona or "").strip() or await self._persona_identity(lanlan, kwargs)
        if not target:
            return Err(SdkError("无法确定当前角色；请显式传 persona/lanlan", code="NO_PERSONA"))
        store = await self._store_for(target)
        cap = max(1, min(int(limit or 12), 24))
        truncated = False

        try:
            if query.strip():
                rows, truncated = await pipeline.search(query, lanlan=target, top_k=cap)
                pairs = list(rows)
            else:
                recent = await asyncio.to_thread(store.recent, lanlan=target, limit=cap)
                pairs = [(row, 0.0) for row in recent]
        except Exception as exc:  # noqa: BLE001
            self._log("error", f"面板查询失败：{type(exc).__name__}: {exc}")
            return Err(SdkError(f"查询失败：{type(exc).__name__}: {exc}", code="BOARD_FAILED"))

        items = []
        for row, score in pairs:
            item = _row_view(row, score)
            if include_thumb:
                item["thumb_data_url"] = await asyncio.to_thread(
                    self._thumb_data_url, row, target
                )
            items.append(item)
        return Ok({"count": len(items), "truncated": truncated, "lanlan": target, "items": items})

    @plugin_entry(
        id="img_log",
        name="运行日志",
        description="最近的活动与错误记录，面板用。",
        input_schema=_LOG_SCHEMA,
        timeout=30.0,
    )
    async def img_log(self, limit: int = 40, level: str = "", persona: str = "", **_):
        _pipeline, archives = await self._require()
        if archives is None:
            return Err(SdkError(self._init_error or "插件尚未就绪", code="NOT_READY"))
        target = (persona or "").strip() or await self._persona_identity()
        if not target:
            return Err(SdkError("无法确定当前角色", code="NO_PERSONA"))
        store = await self._store_for(target)
        try:
            rows = await asyncio.to_thread(
                store.recent_events,
                limit=max(1, min(int(limit or 40), 200)),
                level=level if level in {"info", "warn", "error"} else "",
            )
        except StoreError as exc:
            return Err(SdkError(f"读取日志失败：{exc}", code="LOG_FAILED"))
        return Ok({"count": len(rows), "items": rows, "lanlan": target})

    @plugin_entry(
        id="img_reassign",
        name="改归属",
        description="把图片归到某个角色名下；用于收回「未归属」的旧记录。",
        input_schema=_REASSIGN_SCHEMA,
        timeout=20.0,
    )
    async def img_reassign(self, image_id: int = 0, lanlan: str = "", from_lanlan: str = "", **kwargs):
        _pipeline, archives = await self._require()
        if archives is None:
            return Err(SdkError(self._init_error or "插件尚未就绪", code="NOT_READY"))
        source = (from_lanlan or "").strip() or await self._persona_identity()
        target = (lanlan or "").strip()
        if not source:
            return Err(SdkError("无法确定源角色", code="NO_PERSONA"))
        if not target:
            return Err(
                SdkError("需要目标角色：显式传 lanlan（不会默认搬回当前角色）", code="INVALID_ARGUMENT")
            )
        try:
            new_id = await asyncio.to_thread(
                archives.move,
                from_persona=source,
                image_id=int(image_id or 0),
                to_persona=target,
            )
        except StoreError as exc:
            return Err(SdkError(str(exc), code="REASSIGN_FAILED"))
        await self._event(
            "reassign", f"#{image_id} 迁到 {target}（新 id #{new_id}）", image_id=int(image_id), persona=target
        )
        return Ok({"id": new_id, "from": source, "lanlan": target})

    def _thumb_data_url(self, row: Any, persona: str = "") -> str:
        """缩略图转 data URL。面板拿不到插件数据目录，只能这样喂图。"""
        rel = str(row.thumb_path or row.rel_path or "")
        if not rel:
            return ""
        try:
            path = self._media_file(rel, persona)
        except ValueError:
            return ""
        if not path.is_file():
            return ""
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        if len(data) > 400_000:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")

    def _persona_status(self) -> dict[str, object]:
        persona = self._persona_cache
        if persona is None:
            return {"source": "unknown", "name": "", "chars": 0}
        return {
            "source": persona.source,
            "name": persona.name,
            "chars": len(persona.text),
            "warning": persona.warning,
        }

    # ------------------------------------------------------------------ LLM 工具

    @llm_tool(
        name="image_memory_search",
        description="在本机图像记忆库里按语义检索图片，返回描述与相似度，供你决定要不要发图。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "想找什么样的图"},
                "top_k": {"type": "integer", "description": "返回条数，默认 5"},
            },
            "required": ["query"],
        },
        timeout=30.0,
    )
    async def image_memory_search(self, query: str = "", top_k: int = 5):
        try:
            pipeline, archives = await self._require()
            if pipeline is None or archives is None:
                return {"output": {"reason": self._init_error or "插件尚未就绪"}, "is_error": True,
                        "error": "IMAGE_MEMORY_NOT_READY"}
            if not query.strip():
                return {"output": {"reason": "query 不能为空"}, "is_error": True,
                        "error": "IMAGE_MEMORY_BAD_QUERY"}
            scope = await self._persona_identity()
            if not scope:
                return {"output": {"reason": "无法确定当前角色，已按人设隔离策略拒绝检索"},
                        "is_error": True, "error": "IMAGE_MEMORY_NO_PERSONA"}
            rows, truncated = await pipeline.search(
                query, lanlan=scope, top_k=max(1, min(int(top_k), 20))
            )
            return {"count": len(rows), "truncated": truncated,
                    "results": [_row_view(row, score) for row, score in rows]}
        except Exception as exc:  # noqa: BLE001
            return {"output": {"reason": f"{type(exc).__name__}: {exc}"}, "is_error": True,
                    "error": "IMAGE_MEMORY_SEARCH_FAILED"}

    @llm_tool(
        name="image_memory_send",
        description="检索并把一张图片发到当前聊天里；只在你确实想给对方看图时调用。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "想发什么样的图"},
                "image_id": {"type": "integer", "description": "已知 id 时可直接指定"},
            },
        },
        timeout=30.0,
    )
    async def image_memory_send(self, query: str = "", image_id: int = 0):
        try:
            pipeline, archives = await self._require()
            if pipeline is None or archives is None:
                return {"output": {"reason": self._init_error or "插件尚未就绪"}, "is_error": True,
                        "error": "IMAGE_MEMORY_NOT_READY"}
            lanlan = await self._persona_identity()
            if not lanlan:
                return {"output": {"reason": "无法确定当前角色，已按人设隔离策略拒绝"},
                        "is_error": True, "error": "IMAGE_MEMORY_NO_PERSONA"}
            store = await self._store_for(lanlan)
            row = store.get(int(image_id or 0)) if int(image_id or 0) > 0 else None
            if row is not None and row.lanlan and row.lanlan != lanlan:
                return {"output": {"reason": "这张图不属于当前角色"},
                        "is_error": True, "error": "IMAGE_MEMORY_FORBIDDEN"}
            if row is None:
                if not query.strip():
                    return {"output": {"reason": "query 与 image_id 至少给一个"}, "is_error": True,
                            "error": "IMAGE_MEMORY_BAD_QUERY"}
                rows, _ = await pipeline.search(query, lanlan=lanlan, top_k=1)
                if not rows:
                    return {"output": {"reason": "没有匹配的图片"}, "is_error": True,
                            "error": "IMAGE_MEMORY_EMPTY"}
                row = rows[0][0]
            sent = await self._push_image(row, lanlan or row.lanlan)
            await asyncio.to_thread(store.touch, row.id)
            return sent
        except Exception as exc:  # noqa: BLE001
            return {"output": {"reason": f"{type(exc).__name__}: {exc}"}, "is_error": True,
                    "error": "IMAGE_MEMORY_SEND_FAILED"}

    # ------------------------------------------------------------------ 发图

    async def _push_image(self, row: Any, target: str) -> dict[str, object]:
        path = self._media_file(str(row.rel_path), target)
        if not path.is_file():
            raise FileNotFoundError(f"图片文件缺失：{row.rel_path}")
        data = await asyncio.to_thread(path.read_bytes)

        images = getattr(self.ctx, "images", None)
        if images is None or not hasattr(images, "upload"):
            raise RuntimeError("ctx.images.upload 不可用，无法把图片放进聊天")
        part = await images.upload(data, mime="image/jpeg")

        parts: list[dict[str, object]] = []
        if row.caption:
            parts.append({"type": "text", "text": str(row.caption)})
        parts.append(dict(part))

        push = getattr(self, "push_message", None) or getattr(self.ctx, "push_message", None)
        if push is None:
            raise RuntimeError("push_message 不可用")
        receipt = push(
            visibility=["chat"],
            ai_behavior="blind",
            parts=parts,
            source=self.plugin_id,
            target_lanlan=target or None,
            priority=2,
        )
        if inspect.isawaitable(receipt):
            receipt = await receipt
        submitted, reason = _receipt_state(receipt)
        if not submitted:
            raise RuntimeError(f"消息被拒：{reason}")
        return {"sent": True, "id": int(row.id), "caption": row.caption, "lanlan": target}

    # ------------------------------------------------------------------ 定时器

    @timer_interval(id="img_watch", seconds=30, auto_start=True)
    async def img_watch(self, **_):
        settings = await self._settings_snapshot()
        # 即使没开自动收录，也借这个 tick 把初始化推进到就绪（失败有 60 秒冷却）。
        if not self._ready:
            await self._ensure()
        # 把 openai 的 5 秒冷导入挪到线程里预热一次，别让它落在第一次收录的事件循环上。
        if not self._warm_imports:
            self._warm_imports = True
            with contextlib.suppress(Exception):
                await asyncio.to_thread(ensure_sdk_imported)
        if not settings.watch_frames:
            return Ok({"skipped": "watch_frames=false", "ready": self._ready})
        pipeline, archives = await self._require()
        if pipeline is None or archives is None:
            return Ok({"skipped": self._init_error or "not ready"})
        try:
            data, mime = await self._frame_bytes()
        except ValueError as exc:
            return Ok({"skipped": str(exc)})
        lanlan = await self._persona_identity()
        if not lanlan:
            return Ok({"skipped": "无法确定当前角色"})
        store = await self._store_for(lanlan)
        if store.find_by_sha(self._sha(data), lanlan):
            return Ok({"skipped": "该帧已收录"})
        persona = await self._persona_text()
        result = await pipeline.run(data, mime=mime, lanlan=lanlan, persona_text=persona)
        if result.accepted:
            await self._event(
                "watch",
                f"自动收录：{result.code}",
                image_id=result.image_id,
                engine=(result.verdict.engine if result.verdict else ""),
            )
        return Ok({"accepted": result.accepted, "code": result.code, "id": result.image_id})

    @staticmethod
    def _sha(data: bytes) -> str:
        import hashlib

        return hashlib.sha256(data).hexdigest()


def _row_view(row: Any, score: float) -> dict[str, object]:
    return {
        "id": int(row.id),
        "score": round(float(score), 4),
        "caption": row.caption,
        "tags": list(row.tags),
        "sticker": bool(row.sticker),
        "engine": row.engine,
        "modality": row.modality,
        "value": round(float(row.value), 3),
        "created_at": row.created_at,
        "thumb": row.thumb_path,
        "lanlan": row.lanlan,
    }
