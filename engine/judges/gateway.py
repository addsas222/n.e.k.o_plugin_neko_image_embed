"""gateway 级：走宿主模型网关（``ctx.models``）。

这一级是唯一「真视觉」的判定，但也是本机最容易失败的一级：``[plugin.models.judge]``
必须先在插件面板绑定槽位，否则 403 ``model_usage_not_bound``。所以：

* 这里不缓存 client（SDK 按 event loop 缓存，跨 loop 复用会 RuntimeError）；
* 任何失败都包成 JudgeError，交给链去降级，绝不吞掉；
* ``response_format`` 先试再退——Anthropic 协议的槽位会直接拒绝这个参数。
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time
from dataclasses import replace

from .base import (
    JudgeError,
    JudgeRequest,
    Verdict,
    build_system_prompt,
    build_user_text,
    coerce_verdict,
    parse_json_loose,
)

_PARAM_ERROR_STATUS = {400, 404, 422}
_PARAM_ERROR_TYPES = {"BadRequestError", "UnprocessableEntityError", "NotFoundError"}

_OPENAI_READY = False
_IMPORT_LOCK = threading.Lock()


def ensure_sdk_imported() -> bool:
    """把 ``openai`` 的首次导入挪到线程里。

    这个包在本机冷导入约 5 秒（实测 4.95s）。它是在 ``models.get_client()`` 里懒加载的，
    也就是发生在事件循环上——宿主会直接报 ``Plugin event-loop stall`` 并把整条命令流卡住。
    所有调用点都必须用 ``asyncio.to_thread(ensure_sdk_imported)``。
    """
    global _OPENAI_READY
    if _OPENAI_READY:
        return True
    with _IMPORT_LOCK:
        if _OPENAI_READY:
            return True
        try:
            importlib.import_module("openai")
        except Exception:  # noqa: BLE001 - 缺包时让 get_client 报它自己的错
            return False
        _OPENAI_READY = True
        return True


def _is_param_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _PARAM_ERROR_STATUS:
        return True
    return type(exc).__name__ in _PARAM_ERROR_TYPES


def _is_binding_error(exc: BaseException) -> bool:
    """槽位没绑 / 没声明：这类 403 重试也没用，要冷却而不是每张图都撞一次。"""
    if getattr(exc, "status_code", None) != 403:
        return False
    text = str(exc)
    return "model_usage_not_bound" in text or "model_usage_not_declared" in text


class GatewayJudge:
    engine = "gateway"

    def __init__(self, ctx: object, *, usage: str, timeout: float, max_tokens: int) -> None:
        self._ctx = ctx
        self._usage = usage
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._blocked_until = 0.0

    async def judge(self, request: JudgeRequest) -> Verdict:
        if time.time() < self._blocked_until:
            raise JudgeError(self.engine, "unbound", "模型槽位未绑定，冷却中（跳过以免每张图都等一次 403）")

        models = getattr(self._ctx, "models", None)
        if models is None:
            raise JudgeError(self.engine, "unavailable", "ctx.models 不存在")

        # 冷导入 5 秒，必须在循环之外完成（见 ensure_sdk_imported 的说明）。
        await asyncio.to_thread(ensure_sdk_imported)

        try:
            client = await models.get_client()
        except Exception as exc:  # noqa: BLE001 - 能力缺失/网关未签发都归到这里
            raise JudgeError(self.engine, "client", f"{type(exc).__name__}: {exc}") from exc

        content: list[dict[str, object]] = [{"type": "text", "text": build_user_text(request)}]
        if request.image_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{request.mime};base64,{request.image_b64}",
                        "detail": "low",
                    },
                }
            )

        payload: dict[str, object] = {
            "model": self._usage,
            "messages": [
                {"role": "system", "content": build_system_prompt(request.persona)},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": self._max_tokens,
            "temperature": 0.2,
        }

        try:
            try:
                response = await client.chat.completions.create(
                    **payload, response_format={"type": "json_object"}
                )
            except Exception as exc:  # noqa: BLE001
                if not _is_param_error(exc):
                    raise
                response = await client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001 - 网络/配额/鉴权都归到调用失败
            if _is_binding_error(exc):
                self._blocked_until = time.time() + 300.0
            raise JudgeError(self.engine, "call", f"{type(exc).__name__}: {exc}") from exc

        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise JudgeError(self.engine, "shape", f"应答结构异常: {exc}") from exc

        try:
            verdict = coerce_verdict(
                parse_json_loose(text), engine=self.engine, modality="vision"
            )
        except ValueError as exc:
            raise JudgeError(self.engine, "parse", f"{exc}: {text[:200]}") from exc

        return replace(verdict, raw=text[:1000])
