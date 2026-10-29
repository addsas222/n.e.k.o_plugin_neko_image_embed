"""判定契约：统一 Verdict、严格 JSON 解析、提示词构造。

所有引擎（宿主网关 / OpenAI 兼容 / SystemOne / 规则）都必须收敛到同一个
``Verdict``，否则链上降级会让上层拿到形状不同的结果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..guard import Metrics, Probe

RISK_KEYS = ("nsfw", "violence", "politics", "pi")


class JudgeError(RuntimeError):
    """单个引擎失败。``kind`` 用于区分「没配好」和「调用出错」。"""

    def __init__(self, engine: str, kind: str, detail: str) -> None:
        super().__init__(f"{engine}: {kind}: {detail}")
        self.engine = engine
        self.kind = kind
        self.detail = detail[:500]


@dataclass(frozen=True)
class JudgeRequest:
    persona: str
    digest: str
    image_b64: str = ""
    mime: str = "image/jpeg"
    caption_hint: str = ""
    tags: tuple[str, ...] = ()
    metrics: Metrics | None = None
    probe: Probe | None = None


@dataclass(frozen=True)
class Verdict:
    should_embed: bool
    value: float
    risk: dict[str, float] = field(default_factory=dict)
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    sticker: bool = False
    reason: str = ""
    engine: str = ""
    perception: str = ""
    modality: str = "vision"  # vision | digest | heuristic
    raw: str = ""
    issues: tuple[str, ...] = ()

    @property
    def risk_max(self) -> float:
        return max(self.risk.values(), default=0.0)


def parse_json_loose(text: str) -> dict:
    """从模型输出里挖出第一个平衡的 JSON 对象。

    模型会在前后加解释、```json 围栏、甚至重复输出两个对象；这里不依赖
    ``response_format``（Anthropic 协议的槽位会直接拒绝该参数），所以必须自己扛。
    """
    if not isinstance(text, str):
        raise ValueError("输出不是字符串")
    stripped = text.strip()
    if not stripped:
        raise ValueError("空输出")

    fenced = stripped
    if "```" in fenced:
        parts = fenced.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                fenced = candidate
                break

    start = fenced.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(fenced)):
            char = fenced[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    chunk = fenced[start : index + 1]
                    try:
                        parsed = json.loads(chunk)
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = fenced.find("{", start + 1)
    raise ValueError("没有找到可解析的 JSON 对象")


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "y", "1", "是", "收录", "要", "值得"}:
            return True
        if token in {"false", "no", "n", "0", "否", "不收录", "不要", "不值得"}:
            return False
    return default


def _as_float(value: object, default: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return default
    else:
        return default
    if number != number:
        return default
    return min(max(number, lo), hi)


def _as_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _as_tags(value: object, limit: int = 12) -> list[str]:
    if isinstance(value, str):
        raw = [piece for piece in value.replace("，", ",").split(",")]
    elif isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        return []
    tags: list[str] = []
    for item in raw:
        tag = item.strip()[:24]
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def coerce_verdict(payload: object, *, engine: str, modality: str) -> Verdict:
    """把任意形状的模型输出收成 Verdict，字段级兜底并记录 issues。"""
    issues: list[str] = []
    if not isinstance(payload, dict):
        raise ValueError("判定输出不是对象")

    if "should_embed" not in payload:
        issues.append("缺少 should_embed，按不收录处理")
    should_embed = _as_bool(payload.get("should_embed"), False)

    value_raw = payload.get("value")
    value = _as_float(value_raw, 0.0)
    if value_raw is None:
        issues.append("缺少 value，按 0 处理")

    risk_raw = payload.get("risk")
    risk: dict[str, float] = {}
    if isinstance(risk_raw, dict):
        for key in RISK_KEYS:
            if key in risk_raw:
                risk[key] = _as_float(risk_raw.get(key), 0.0)
    elif risk_raw is not None:
        issues.append("risk 不是对象，已忽略")

    return Verdict(
        should_embed=should_embed,
        value=value,
        risk=risk,
        caption=_as_text(payload.get("caption"), 300),
        tags=_as_tags(payload.get("tags")),
        sticker=_as_bool(payload.get("sticker"), False),
        reason=_as_text(payload.get("reason"), 240),
        engine=engine,
        modality=modality,
        issues=tuple(issues),
    )


SYSTEM_PROMPT = """你是一个「图像记忆收录判定器」。你要判断：给定角色的设定，这张图是否值得\
被这个角色记住，并在以后聊天时可能被想起来或发出去。

判定原则：
1. 只有和角色兴趣、关系、日常场景真正相关的图才值得记；纯截图、表格、无意义涂鸦不值得。
2. 表情包/梗图如果和角色的说话风格契合，sticker 记 true。
3. 不确定时倾向不收录（should_embed=false），不要为了凑数放宽。
4. 风险维度只在有明确迹象时给高分，不要臆测。

只输出一个 JSON 对象，不要解释、不要 markdown 围栏：
{"should_embed": true/false, "value": 0.0-1.0, "risk": {"nsfw":0.0,"violence":0.0,"politics":0.0,"pi":0.0}, "caption": "一句话描述画面", "tags": ["标签"], "sticker": true/false, "reason": "为什么值得/不值得记"}
"""


def build_system_prompt(persona_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n当前角色设定：\n{persona_text or '（未知角色，按通用陪伴角色判断）'}\n"


def build_user_text(request: JudgeRequest) -> str:
    lines = ["请判断这张图是否值得收录进该角色的图像记忆。", f"图像指标：{request.digest}"]
    if request.caption_hint:
        lines.append(f"本地预判描述：{request.caption_hint}")
    return "\n".join(lines)
