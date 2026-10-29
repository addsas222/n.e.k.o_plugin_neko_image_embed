"""配置收敛：把宿主给的原始 dict 变成不可变 Settings。

三条约束，全部来自 SDK 的实际行为：

* 配置后端可能整体失败（IPC 超时/未就绪），调用方必须能在「没有任何配置」时继续跑，
  所以 ``load_settings({})`` 必须得出全套默认值而不是抛异常。
* 单个字段类型错误不该拖垮整份配置：只回落该字段默认值，并留一条 warning。
* 本模块不 import ``plugin.sdk``、不读写文件，保持纯函数，方便脱离宿主单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_ENGINE_ORDER: tuple[str, ...] = ("laya", "gateway", "rules")
DEFAULT_PERCEPTION: tuple[str, ...] = ("sensenova", "zeroshot")
JUDGE_MODES: tuple[str, ...] = ("model", "host", "heuristic")
KNOWN_ENGINES: tuple[str, ...] = (
    "gateway", "sensenova", "model", "laya", "kev", "semif", "jev",
)
LOCAL_ENGINES: tuple[str, ...] = ("zeroshot", "rules")
VALID_PROTOCOLS = ("openai_chat", "systemone")


@dataclass(frozen=True)
class Limits:
    max_source_bytes: int = 32 * 1024 * 1024
    max_source_pixels: int = 16 * 1024 * 1024
    min_sharpness: float = 40.0
    min_entropy: float = 2.0
    min_value: float = 0.45
    max_risk: float = 0.60
    dup_hamming_max: int = 6


@dataclass(frozen=True)
class Storage:
    long_edge: int = 1280
    jpeg_quality: int = 82
    max_bytes: int = 512 * 1024
    thumb_edge: int = 256
    thumb_quality: int = 75
    allow_animated: bool = False
    default_top_k: int = 8
    max_search_k: int = 50
    max_scan_rows: int = 20000


@dataclass(frozen=True)
class EmbedderConfig:
    backend: str = "auto"
    model_dir: str = ""
    hf_endpoint: str = "https://hf-mirror.com"
    allow_download: bool = True
    external_base_url: str = ""
    external_api_key: str = ""
    external_model: str = ""
    external_dim: int = 0


@dataclass(frozen=True)
class EngineConfig:
    id: str
    enabled: bool = False
    protocol: str = "openai_chat"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    vision: bool = True

    def ready(self) -> bool:
        """配置层面是否可用（不探活；联网与否由调用处判定）。"""
        if not self.enabled:
            return False
        if self.id == "gateway":
            return True
        return bool(self.base_url.strip())


@dataclass(frozen=True)
class JudgeConfig:
    mode: str = "heuristic"
    perception: tuple[str, ...] = DEFAULT_PERCEPTION
    order: tuple[str, ...] = DEFAULT_ENGINE_ORDER
    timeout_seconds: float = 60.0
    max_completion_tokens: int = 700
    engines: dict[str, EngineConfig] = field(default_factory=dict)

    def chain(self) -> list[EngineConfig]:
        """按 order 展开 HTTP 引擎；gateway 由上层拿 ctx 单独装配。"""
        out: list[EngineConfig] = []
        for name in self.order:
            if name == "rules" or name == "zeroshot":
                continue
            engine = self.engines.get(name)
            if engine is not None and engine.ready():
                out.append(engine)
        return out


@dataclass(frozen=True)
class Settings:
    limits: Limits = field(default_factory=Limits)
    storage: Storage = field(default_factory=Storage)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    persona_summary: str = ""
    characters_path: str = ""
    watch_frames: bool = False
    watch_interval_seconds: int = 30
    query_aliases: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------- 取值助手


class _Reader:
    def __init__(self, warnings: list[str]) -> None:
        self.warnings = warnings

    def _warn(self, key: str, want: str, value: object) -> None:
        self.warnings.append(f"{key}: 期望 {want}，得到 {value!r}，已回落默认值")

    def int(self, section: dict[str, Any], key: str, default: int, *, lo: int, hi: int) -> int:
        raw = section.get(key, default)
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            self._warn(key, "整数", raw)
            return default
        if not lo <= value <= hi:
            self.warnings.append(f"{key}: {value} 超出 [{lo}, {hi}]，已夹到边界")
            return min(max(value, lo), hi)
        return value

    def float(self, section: dict[str, Any], key: str, default: float, *, lo: float, hi: float) -> float:
        raw = section.get(key, default)
        if raw is None:
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            self._warn(key, "数字", raw)
            return default
        if value != value or value in (float("inf"), float("-inf")):
            self._warn(key, "有限数字", raw)
            return default
        if not lo <= value <= hi:
            self.warnings.append(f"{key}: {value} 超出 [{lo}, {hi}]，已夹到边界")
            return min(max(value, lo), hi)
        return value

    def bool(self, section: dict[str, Any], key: str, default: bool) -> bool:
        raw = section.get(key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
            return raw.strip().lower() == "true"
        if isinstance(raw, int) and raw in (0, 1):
            return bool(raw)
        if raw is None:
            return default
        self._warn(key, "布尔值", raw)
        return default

    def text(self, section: dict[str, Any], key: str, default: str = "") -> str:
        raw = section.get(key, default)
        if raw is None:
            return default
        if isinstance(raw, str):
            return raw.strip()
        self.warnings.append(f"{key}: 期望字符串，得到 {type(raw).__name__}，已转成字符串")
        return str(raw).strip()

    def mapping(self, section: dict[str, Any], key: str) -> dict[str, str]:
        raw = section.get(key)
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            self.warnings.append(f"{key}: 期望表（键值都是字符串），已忽略")
            return {}
        out: dict[str, str] = {}
        for name, value in raw.items():
            if isinstance(name, str) and isinstance(value, str) and name.strip() and value.strip():
                out[name.strip()] = value.strip()
            else:
                self.warnings.append(f"{key}.{name}: 需要非空字符串，已忽略")
        return out


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


def _parse_engine(engine_id: str, raw: dict[str, Any], reader: _Reader) -> EngineConfig:
    protocol = reader.text(raw, "protocol", "systemone" if engine_id in {"kev", "jev"} else "openai_chat")
    if protocol not in VALID_PROTOCOLS:
        reader.warnings.append(f"judge.{engine_id}.protocol: {protocol!r} 不支持，回落 openai_chat")
        protocol = "openai_chat"
    return EngineConfig(
        id=engine_id,
        enabled=reader.bool(raw, "enabled", engine_id == "gateway"),
        protocol=protocol,
        base_url=reader.text(raw, "base_url"),
        api_key=reader.text(raw, "api_key"),
        model=reader.text(raw, "model"),
        vision=reader.bool(raw, "vision", protocol == "openai_chat"),
    )


def load_settings(raw: Any) -> Settings:
    """任何输入都必须返回一份可用的 Settings，绝不抛出。"""
    warnings: list[str] = []
    reader = _Reader(warnings)
    root: dict[str, Any] = raw if isinstance(raw, dict) else {}
    if not isinstance(raw, dict):
        warnings.append("配置根不是对象，已全部使用默认值")

    section = _section(root, "image_embed")

    limits = Limits(
        max_source_bytes=reader.int(section, "max_source_bytes", 32 * 1024 * 1024, lo=1024, hi=256 * 1024 * 1024),
        max_source_pixels=reader.int(section, "max_source_pixels", 16 * 1024 * 1024, lo=4096, hi=512 * 1024 * 1024),
        min_sharpness=reader.float(section, "min_sharpness", 40.0, lo=0.0, hi=100000.0),
        min_entropy=reader.float(section, "min_entropy", 2.0, lo=0.0, hi=8.0),
        min_value=reader.float(section, "min_value", 0.45, lo=0.0, hi=1.0),
        max_risk=reader.float(section, "max_risk", 0.60, lo=0.0, hi=1.0),
        dup_hamming_max=reader.int(section, "dup_hamming_max", 6, lo=0, hi=64),
    )

    storage = Storage(
        long_edge=reader.int(section, "store_long_edge", 1280, lo=64, hi=4096),
        jpeg_quality=reader.int(section, "store_jpeg_quality", 82, lo=40, hi=95),
        max_bytes=reader.int(section, "store_max_bytes", 512 * 1024, lo=16 * 1024, hi=8 * 1024 * 1024),
        thumb_edge=reader.int(section, "thumb_edge", 256, lo=32, hi=1024),
        thumb_quality=reader.int(section, "thumb_jpeg_quality", 75, lo=30, hi=95),
        allow_animated=reader.bool(section, "allow_animated", False),
        default_top_k=reader.int(section, "default_top_k", 8, lo=1, hi=50),
        max_search_k=reader.int(section, "max_search_k", 50, lo=1, hi=200),
        max_scan_rows=reader.int(section, "max_scan_rows", 20000, lo=100, hi=500000),
    )

    emb_raw = _section(section, "embedder")
    backend = reader.text(emb_raw, "backend", "auto") or "auto"
    if backend not in {"auto", "clip_onnx", "caption_text", "external"}:
        warnings.append(f"embedder.backend: {backend!r} 不支持，回落 auto")
        backend = "auto"
    embedder = EmbedderConfig(
        backend=backend,
        model_dir=reader.text(emb_raw, "model_dir"),
        hf_endpoint=reader.text(emb_raw, "hf_endpoint", "https://hf-mirror.com"),
        allow_download=reader.bool(emb_raw, "allow_download", True),
        external_base_url=reader.text(emb_raw, "external_base_url"),
        external_api_key=reader.text(emb_raw, "external_api_key"),
        external_model=reader.text(emb_raw, "external_model"),
        external_dim=reader.int(emb_raw, "external_dim", 0, lo=0, hi=8192),
    )

    judge_raw = _section(section, "judge")
    perception_raw = judge_raw.get("perception", list(DEFAULT_PERCEPTION))
    perception: list[str] = []
    if isinstance(perception_raw, (list, tuple)):
        for item in perception_raw:
            name = str(item).strip()
            if name in ("sensenova", "zeroshot") and name not in perception:
                perception.append(name)
            elif name:
                warnings.append(f"judge.perception: 忽略未知识图引擎 {name!r}")
    if not perception:
        perception = list(DEFAULT_PERCEPTION)

    mode = str(judge_raw.get("mode", "")).strip().lower()
    if mode not in JUDGE_MODES:
        if mode:
            warnings.append(f"judge.mode: 未知取值 {mode!r}，回落到启发式")
        # 兼容旧配置：按已启用的引擎推断
        order_hint = [str(x) for x in (judge_raw.get("order") or [])]
        engine_hint = _section(judge_raw, "laya")
        if any(name in order_hint for name in ("laya", "kev", "semif", "jev", "model")):
            mode = "model"
        elif "gateway" in order_hint:
            mode = "host"
        else:
            mode = "heuristic"
        _ = engine_hint

    order_raw = judge_raw.get("order", list(DEFAULT_ENGINE_ORDER))
    order: list[str] = []
    if isinstance(order_raw, (list, tuple)):
        for item in order_raw:
            name = str(item).strip()
            if name in KNOWN_ENGINES or name in LOCAL_ENGINES:
                if name not in order:
                    order.append(name)
            else:
                warnings.append(f"judge.order: 忽略未知引擎 {name!r}")
    else:
        warnings.append("judge.order: 期望数组，已回落到默认顺序")
    if "rules" not in order:
        order.append("rules")

    engines = {
        name: _parse_engine(name, _section(judge_raw, name), reader)
        for name in KNOWN_ENGINES
    }

    judge = JudgeConfig(
        mode=mode,
        perception=tuple(perception),
        order=tuple(order),
        timeout_seconds=reader.float(judge_raw, "timeout_seconds", 60.0, lo=1.0, hi=300.0),
        max_completion_tokens=reader.int(judge_raw, "max_completion_tokens", 700, lo=64, hi=8192),
        engines=engines,
    )

    return Settings(
        limits=limits,
        storage=storage,
        embedder=embedder,
        judge=judge,
        persona_summary=reader.text(section, "persona_summary"),
        characters_path=reader.text(section, "characters_path"),
        watch_frames=reader.bool(section, "watch_frames", False),
        watch_interval_seconds=reader.int(section, "watch_interval_seconds", 30, lo=5, hi=3600),
        query_aliases=reader.mapping(section, "query_aliases"),
        warnings=tuple(warnings),
    )
