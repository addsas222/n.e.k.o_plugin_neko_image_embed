"""桩 SDK 端到端：不开宿主，把真实插件源码跑一遍全流程。

跑法：
    python tests/stub_run.py

它做的事：
1. 造一棵 ``plugin.sdk.plugin`` shim（NekoPluginBase / 装饰器 / Ok / Err / SdkError）；
2. 以 ``plugin.plugins.neko_image_embed`` 的名字加载**真实** ``__init__.py``；
3. 只替换编码器（不加载 153MB ONNX），其余全用真代码：闸门、判定链、压缩、SQLite；
4. 依次驱动 startup → ingest → 去重 → status → search → send → 遗忘 → gc。

退出码 0 = 全过。任何一项失败会打印 [FAIL] 并返回 1。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import io
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PLUGIN_ROOT / "cache" / "test-tmp"
DIM = 32


# --------------------------------------------------------------------------- 桩 SDK


class SdkError(Exception):
    """与真实 SDK 同形：message + code + details。"""

    def __init__(self, message: str = "", *, code=None, details=None, **_context):
        super().__init__(message)
        self.code = code
        self.details = details


class Ok:
    def __init__(self, value=None):
        self.value = value


class Err:
    def __init__(self, error=None):
        self.error = error


def _decorator(*_args, **_kwargs):
    def wrap(func):
        return func

    return wrap


def _neko_plugin(cls):
    return cls


def _install_stub_sdk() -> None:
    plugin_pkg = types.ModuleType("plugin")
    plugin_pkg.__path__ = []  # type: ignore[attr-defined]
    sdk_pkg = types.ModuleType("plugin.sdk")
    sdk_pkg.__path__ = []  # type: ignore[attr-defined]
    sdk_api = types.ModuleType("plugin.sdk.plugin")
    sdk_api.NekoPluginBase = NekoPluginBase
    sdk_api.neko_plugin = _neko_plugin
    sdk_api.plugin_entry = _decorator
    sdk_api.lifecycle = _decorator
    sdk_api.llm_tool = _decorator
    sdk_api.timer_interval = _decorator
    sdk_api.Ok = Ok
    sdk_api.Err = Err
    sdk_api.SdkError = SdkError

    plugins_pkg = types.ModuleType("plugin.plugins")
    plugins_pkg.__path__ = [str(PLUGIN_ROOT.parent)]  # type: ignore[attr-defined]

    sys.modules["plugin"] = plugin_pkg
    sys.modules["plugin.sdk"] = sdk_pkg
    sys.modules["plugin.sdk.plugin"] = sdk_api
    sys.modules["plugin.plugins"] = plugins_pkg


class NekoPluginBase:
    def __init__(self, ctx):
        self.ctx = ctx
        self.config = ctx.config
        self.logger = ctx.logger
        self.plugin_id = ctx.plugin_id
        self.plugin_dir = ctx.plugin_dir
        self.storage_dir = ctx.storage_dir

    def data_path(self, *parts) -> Path:
        return self.storage_dir.joinpath("data", *parts)

    def cache_path(self, *parts) -> Path:
        return self.storage_dir.joinpath("cache", *parts)

    def report_status(self, payload) -> None:
        self.ctx.status.append(payload)

    def push_message(self, **kwargs):
        self.ctx.pushes.append(kwargs)
        return {"submitted": True}


# --------------------------------------------------------------------------- 假上下文


class FakeConfig:
    def __init__(self, raw: dict):
        self._raw = raw

    async def dump(self, timeout: float | None = None):
        return self._raw


class FakeModels:
    async def get_client(self):
        raise RuntimeError("stub: model gateway 未绑定槽位")


class FakeFrames:
    def __init__(self, records):
        self._records = records

    def get(self, **_kwargs):
        return self._records


class FakeBus:
    def __init__(self, records):
        self.frames = FakeFrames(records)


class FakeImages:
    def __init__(self):
        self.uploads = 0

    async def upload(self, data: bytes, *, mime: str | None = None, timeout: float = 3.0):
        self.uploads += 1
        return {"type": "image", "url": f"http://127.0.0.1:1/media/{self.uploads}", "mime": "image/jpeg"}


class FakeCtx:
    def __init__(self, root: Path, records, raw_config: dict):
        self.plugin_id = "neko_image_embed"
        self.plugin_dir = PLUGIN_ROOT
        self.storage_dir = root / "plugins" / "neko_image_embed"
        self.config = FakeConfig(raw_config)
        self.logger = _Logger()
        self.models = FakeModels()
        self.bus = FakeBus(records)
        self.images = FakeImages()
        self.current_lanlan = "YUI"
        self.status: list[dict] = []
        self.pushes: list[dict] = []


class _Logger:
    def __init__(self):
        self.lines: list[str] = []

    def _emit(self, level: str, message: str) -> None:
        self.lines.append(f"{level}: {message}")

    def info(self, message: str) -> None:
        self._emit("info", str(message))

    def warning(self, message: str) -> None:
        self._emit("warning", str(message))

    def error(self, message: str) -> None:
        self._emit("error", str(message))


# --------------------------------------------------------------------------- 假编码器


class FakeState:
    backend = "fake"
    ready = True
    detail = "stub embedder"
    id = "fake-embedder"
    dim = DIM


class FakeRegistry:
    def __init__(self, *_args, **_kwargs):
        pass

    @property
    def state(self):
        return FakeState()

    @property
    def id(self) -> str:
        return FakeState.id

    @property
    def dim(self) -> int:
        return DIM

    def model_dir(self) -> Path:
        return PLUGIN_ROOT / "models"

    async def prepare(self):
        return FakeState()

    async def embed_image(self, jpeg: bytes, *, caption: str = ""):
        digest = hashlib.sha256(jpeg).digest()
        vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        return vector / np.linalg.norm(vector), "fake"

    async def embed_text(self, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        return vector / np.linalg.norm(vector)


# --------------------------------------------------------------------------- 工具


def _load_plugin_module():
    name = "plugin.plugins.neko_image_embed"
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_ROOT / "__init__.py", submodule_search_locations=[str(PLUGIN_ROOT)]
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jpeg(seed: int = 1, size: tuple[int, int] = (800, 600)) -> bytes:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, "RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


CHECKS: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    CHECKS.append((bool(ok), label, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f"  -> {detail}" if detail else ""))


# --------------------------------------------------------------------------- 主流程


async def run() -> int:
    tmp = TMP_ROOT / f"stub-{uuid.uuid4().hex[:10]}"
    tmp.mkdir(parents=True, exist_ok=False)
    data = _jpeg(seed=1)
    record = SimpleNamespace(
        image_base64=base64.b64encode(_jpeg(seed=2)).decode("ascii"),
        mime="image/jpeg",
        source="user",
        captured_at=1.0,
        frame_id="frame-1",
    )
    ctx = FakeCtx(tmp, [record], {})
    _install_stub_sdk()
    module = _load_plugin_module()
    module.EmbeddingRegistry = FakeRegistry
    # 别为 openai 的 5 秒冷导入付测试时间：桩里直接短路。
    from plugin.plugins.neko_image_embed.engine.judges import gateway as gateway_mod

    gateway_mod.ensure_sdk_imported = lambda: True

    plugin = module.NekoImageEmbedPlugin(ctx)
    started = await plugin.on_startup()
    check(isinstance(started, Ok), "startup 返回 Ok")
    check((started.value or {}).get("status") in {"ready", "deferred"}, "startup 给出明确状态",
          str((started.value or {}).get("status")))
    if not plugin._ready:
        await plugin._ensure()
    check(plugin._ready is True, "初始化后编码器+索引就绪", plugin._init_error)

    first = await plugin.img_ingest(
        source="b64", b64=base64.b64encode(data).decode("ascii"), lanlan="YUI"
    )
    ok = isinstance(first, Ok) and first.value["accepted"]
    check(ok, "收录一张图", "" if not ok else str(first.value.get("code")))

    payload = first.value if isinstance(first, Ok) else {}
    image_id = payload.get("id", 0)
    check(
        payload.get("verdict", {}).get("perception") == "zeroshot",
        "识图走本地零样本",
        str(payload.get("verdict", {}).get("perception")),
    )
    check(
        payload.get("verdict", {}).get("engine") == "rules",
        "判断层是启发式（未配置模型/槽位）",
        str(payload.get("verdict", {}).get("engine")),
    )
    check(isinstance(payload.get("verdict", {}).get("tags"), list), "识图产出 tags 字段")

    second = await plugin.img_ingest(
        source="b64", b64=base64.b64encode(data).decode("ascii"), lanlan="YUI"
    )
    check(
        isinstance(second, Ok) and second.value.get("duplicate") and second.value["id"] == image_id,
        "重复图不新增记录",
    )

    board = await plugin.img_board(limit=5)
    board_items = (board.value or {}).get("items") if isinstance(board, Ok) else []
    check(
        bool(board_items) and board_items[0].get("lanlan") == "YUI",
        "入库图片归属到当前角色",
        str(board_items[0].get("lanlan") if board_items else None),
    )

    original_lanlan = ctx.current_lanlan
    ctx.current_lanlan = ""
    denied = await plugin.img_ingest(
        source="b64", b64=base64.b64encode(data).decode("ascii")
    )
    check(
        isinstance(denied, Err) and getattr(denied.error, "code", "") == "NO_PERSONA",
        "无法确定角色时拒绝入库（不落公共桶）",
    )
    denied_tool = await plugin.image_memory_search(query="狗")
    check(
        denied_tool.get("is_error") is True and denied_tool.get("error") == "IMAGE_MEMORY_NO_PERSONA",
        "无法确定角色时 LLM 工具拒绝检索",
    )
    ctx.current_lanlan = original_lanlan

    status = await plugin.img_status()
    counts = (status.value or {}).get("counts") or {}
    check(isinstance(status, Ok) and status.value["state"] == "ready", "status 报 ready")
    check(counts.get("live") == 1, "索引里有 1 条", str(counts))

    search = await plugin.img_search(query="猫", lanlan="YUI", top_k=5)
    check(isinstance(search, Ok) and search.value["count"] >= 1, "文字检索有结果")

    sent = await plugin.img_send(image_id=image_id)
    check(isinstance(sent, Ok) and sent.value.get("sent") is True, "图片推进聊天")
    parts = ctx.pushes[-1]["parts"] if ctx.pushes else []
    check(any(part.get("type") == "image" for part in parts), "push 的一条里带 image part")
    check(ctx.pushes[-1].get("ai_behavior") == "blind", "发图用 blind，避免重复注入")
    check(ctx.pushes[-1].get("target_lanlan") == "YUI", "target_lanlan 指向当前角色")

    frames = await plugin.img_ingest(source="frames", lanlan="YUI")
    check(isinstance(frames, Ok) and frames.value["accepted"], "从 bus.frames 收录")
    check((status.value or {}).get("judge", {}).get("active") is not None, "status 带判定链信息")

    tool = await plugin.image_memory_search(query="随便", top_k=3)
    check(isinstance(tool, dict) and "results" in tool or tool.get("is_error") is True,
          "LLM 检索工具返回合法信封", str(list(tool)[:3]))

    forgotten = await plugin.img_forget(image_id=image_id)
    check(isinstance(forgotten, Ok), "软删除成功")
    after = await plugin.img_status()
    check(((after.value or {}).get("counts") or {}).get("live") == 1, "软删后仍剩 1 条（frames 那张）")

    gc = await plugin.img_gc(days=0, limit=10)
    check(isinstance(gc, Ok) and gc.value["removed"] >= 1, "gc 清掉墓碑", str(gc.value if isinstance(gc, Ok) else gc))

    empty = await plugin.img_search(query="", lanlan="YUI")
    check(
        isinstance(empty, Err) and getattr(empty.error, "code", "") == "INVALID_ARGUMENT",
        "空查询被拒且错误码是 INVALID_ARGUMENT",
        str(getattr(empty, "error", None)),
    )

    bad_path = await plugin.img_ingest(source="path", path="D:/definitely-not-here.jpg")
    check(
        isinstance(bad_path, Err) and getattr(bad_path.error, "code", "") == "INVALID_ARGUMENT",
        "坏路径报 INVALID_ARGUMENT（不是 INTERNAL）",
    )

    missing = await plugin.img_forget(image_id=99999)
    check(
        isinstance(missing, Err) and getattr(missing.error, "code", "") == "NOT_FOUND",
        "不存在的 id 报 NOT_FOUND",
    )

    log = await plugin.img_log(limit=20)
    rows = (log.value or {}).get("items") if isinstance(log, Ok) else []
    check(bool(rows), "运行日志记录了刚才的操作", f"{len(rows or [])} 条")
    check(
        any(str(row.get("kind")) == "ingest" for row in (rows or [])),
        "日志里有 ingest 记录",
    )

    stopped = await plugin.on_shutdown()
    check(isinstance(stopped, Ok), "shutdown 返回 Ok")

    # 刻意不删 tmp：本机 safe-delete 钩子会把删除动作搬去不可写的全局目录，
    # 反而让一次全过的自检以非零码退出。目录留在 .pytest_tmp/ 下，随时可手删。
    return 0 if all(item[0] for item in CHECKS) else 1


def main() -> int:
    try:
        return asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"[FAIL] 未捕获异常: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
