"""当前人设：配置优先，缺省读宿主 characters.json（只读，绝不写回）。

为什么不用 SDK：插件没有 persona API，只有 ``ctx.current_lanlan``（一个名字）和
``{LANLAN_NAME}``/``{MASTER_NAME}`` 占位符。判定需要「性格/喜好」这类文本，
只能从宿主的角色库里取，或者由用户在插件配置里写死一份摘要。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_FIELDS = ("昵称", "性别", "年龄", "种族", "自称", "核心特质", "行为特点", "厌恶", "一句话台词")
_MAX_CHARS = 700


@dataclass(frozen=True)
class Persona:
    name: str = ""
    text: str = ""
    source: str = "none"  # config | characters | none
    warning: str = ""

    def prompt_block(self) -> str:
        if not self.text:
            return "（没有人设信息，按「通用陪伴角色」判断）"
        return self.text


def default_characters_path(host_root: Path) -> Path:
    return host_root / "config" / "characters.json"


def load_persona(
    *,
    override: str,
    characters_path: str | Path | None,
    max_chars: int = _MAX_CHARS,
) -> Persona:
    """任何失败都退化成「没有人设」，不阻塞收录。"""
    text = (override or "").strip()
    if text:
        name = text.splitlines()[0][:40]
        return Persona(name=name, text=_clip(text, max_chars), source="config")

    if characters_path is None:
        return Persona(warning="未提供 characters.json 路径")
    path = Path(characters_path)
    if not path.is_file():
        return Persona(warning=f"角色库不存在: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Persona(warning=f"角色库读取失败: {type(exc).__name__}: {exc}")
    if not isinstance(raw, dict):
        return Persona(warning="角色库根节点不是对象")

    current = str(raw.get("当前猫娘") or "").strip()
    catgirls = raw.get("猫娘")
    profile: dict = {}
    if isinstance(catgirls, dict) and current:
        candidate = catgirls.get(current)
        if isinstance(candidate, dict):
            profile = candidate
    if not profile and isinstance(catgirls, dict):
        # 没标当前角色时，取第一个带昵称的，避免整段判定失去人设
        for key, value in catgirls.items():
            if isinstance(value, dict) and value.get("昵称"):
                current, profile = str(key), value
                break
    if not profile:
        return Persona(warning="角色库里没有可用的猫娘档案")

    name = str(profile.get("昵称") or current).strip()
    lines = [f"角色：{name}"]
    for field in _FIELDS:
        value = profile.get(field)
        if isinstance(value, str) and value.strip():
            lines.append(f"{field}：{value.strip()}")

    master = raw.get("主人")
    if isinstance(master, dict):
        master_name = str(master.get("昵称") or "").strip()
        if master_name:
            lines.insert(0, f"主人：{master_name}")

    return Persona(name=name, text=_clip("\n".join(lines), max_chars), source="characters")


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
