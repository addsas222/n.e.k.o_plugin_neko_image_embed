"""laya / semif（OpenAI 兼容）与 kev / jev（SystemOne）两个执行器。

为什么合并成一个类：它们只差「请求怎么拼、应答怎么读」，失败语义完全一致
（非 2xx、超时、非 JSON、缺字段都算这一级失败）。分开写会复制四份降级逻辑。

只用标准库 urllib，和 ``game_decision/engine/systemone.py`` 同一套做法；
同步调用整体丢进 ``asyncio.to_thread``，绝不阻塞插件的命令循环。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import urllib.error
import urllib.request

from ..config import EngineConfig
from .base import (
    JudgeError,
    JudgeRequest,
    Verdict,
    build_system_prompt,
    build_user_text,
    coerce_verdict,
    parse_json_loose,
)

_SCORE_LEVELS = ["毫无价值", "偏低", "一般", "偏高", "很高"]


class _HttpStatusError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _join(base: str, suffix: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return base + "/" + suffix
    return base + "/v1/" + suffix


class HttpJudge:
    """一个 EngineConfig 一个实例。"""

    def __init__(self, engine: EngineConfig, *, timeout: float, max_tokens: int) -> None:
        self.engine = engine.id
        self._cfg = engine
        self._timeout = max(1.0, float(timeout))
        self._max_tokens = max_tokens

    async def judge(self, request: JudgeRequest) -> Verdict:
        return await asyncio.to_thread(self._judge_sync, request)

    # ------------------------------------------------------------------ 同步实现

    def _judge_sync(self, request: JudgeRequest) -> Verdict:
        if self._cfg.protocol == "systemone":
            return self._systemone(request)
        return self._openai(request)

    def _openai(self, request: JudgeRequest) -> Verdict:
        url = _join(self._cfg.base_url, "chat/completions")
        text_content = build_user_text(request)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": build_system_prompt(request.persona)},
            {"role": "user", "content": text_content},
        ]

        if self._cfg.vision and request.image_b64:
            messages[1] = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{request.mime};base64,{request.image_b64}",
                            "detail": "low",
                        },
                    },
                ],
            }

        payload = {
            "model": self._cfg.model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": 0.2,
        }

        try:
            result = self._post(url, payload)
        except _HttpStatusError as exc:
            # 端点不认图片时退回纯文本 digest，而不是整级判死。
            if not (self._cfg.vision and request.image_b64) or exc.status not in {400, 415, 422}:
                raise JudgeError(self.engine, "http", str(exc)) from exc
            payload["messages"] = [
                {"role": "system", "content": build_system_prompt(request.persona)},
                {"role": "user", "content": text_content},
            ]
            try:
                result = self._post(url, payload)
            except _HttpStatusError as retry_exc:
                raise JudgeError(self.engine, "http", str(retry_exc)) from retry_exc

        try:
            text = result["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError(self.engine, "shape", f"应答结构异常: {exc}") from exc

        try:
            verdict = coerce_verdict(
                parse_json_loose(text), engine=self.engine, modality="vision"
            )
        except ValueError as exc:
            raise JudgeError(self.engine, "parse", f"{exc}: {text[:200]}") from exc
        return Verdict(
            should_embed=verdict.should_embed,
            value=verdict.value,
            risk=verdict.risk,
            caption=verdict.caption,
            tags=verdict.tags,
            sticker=verdict.sticker,
            reason=verdict.reason,
            engine=self.engine,
            modality="vision" if self._cfg.vision and request.image_b64 else "digest",
            raw=text[:1000],
            issues=verdict.issues,
        )

    def _systemone(self, request: JudgeRequest) -> Verdict:
        url = _join(self._cfg.base_url, "systemone")
        state = "\n".join(
            part
            for part in (
                request.persona,
                f"图像指标：{request.digest}",
                request.caption_hint and f"本地预判描述：{request.caption_hint}",
            )
            if part
        )
        payload: dict[str, object] = {
            "state": state,
            "questions": {
                "should_embed": {
                    "type": "choice",
                    "instructions": "结合角色设定，这张图是否值得被记住？",
                    "criteria": {
                        "收录": "与角色兴趣/关系/日常相关，值得记住",
                        "不收录": "无关、无意义或质量太差",
                    },
                },
                "value": {
                    "type": "score",
                    "instructions": "这张图作为角色记忆的价值有多高？",
                    "criteria": _SCORE_LEVELS,
                },
                "sticker": {
                    "type": "choice",
                    "instructions": "这张图适合当表情包在聊天里发出去吗？",
                    "criteria": {"是": "适合", "否": "不适合"},
                },
            },
        }
        if self._cfg.model:
            payload["model"] = self._cfg.model

        try:
            result = self._post(url, payload)
        except _HttpStatusError as exc:
            raise JudgeError(self.engine, "http", str(exc)) from exc

        answers = result.get("answers")
        normalized = _normalize_answers(answers)
        if not normalized:
            raise JudgeError(self.engine, "shape", "应答缺少可解析的 answers")

        should = _choice_to_bool(normalized.get("should_embed"))
        value = _score_to_unit(normalized.get("value"))
        # SystemOne（kev/jev）会给出选项概率；它比 5 档 score 更能反映置信度，
        # 取两者较大值，避免「收录但 score=1.7/4」被 min_value 误杀。
        choice = normalized.get("should_embed")
        if isinstance(choice, dict):
            probs = choice.get("probabilities")
            if isinstance(probs, dict):
                p_yes = probs.get("收录", probs.get("是"))
                if p_yes is None:
                    neg = sum(
                        float(v) for k, v in probs.items() if k in ("不收录", "否")
                    )
                    p_yes = max(0.0, 1.0 - neg) if probs else None
                if p_yes is not None:
                    # 用「收录」这一项的概率作为价值，避免把「不收录」的高置信度当成高分
                    value = float(p_yes)
        sticker = _choice_to_bool(normalized.get("sticker"))
        if "should_embed" not in normalized:
            raise JudgeError(self.engine, "shape", "answers 里没有 should_embed")

        return Verdict(
            should_embed=should,
            value=value,
            risk={},
            caption="",
            tags=[],
            sticker=sticker,
            reason=f"SystemOne 判定：{'收录' if should else '不收录'}（价值 {value:.2f}）",
            engine=self.engine,
            modality="digest",
            raw=json.dumps(normalized, ensure_ascii=False)[:1000],
        )

    def _post(self, url: str, payload: dict[str, object]) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self._cfg.api_key:
            headers["authorization"] = f"Bearer {self._cfg.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            with contextlib.suppress(Exception):
                detail = exc.read(300).decode("utf-8", "replace")
            raise _HttpStatusError(int(exc.code), detail) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise JudgeError(self.engine, "network", f"连接失败: {exc}") from exc

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise JudgeError(self.engine, "parse", f"应答不是 JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise JudgeError(self.engine, "shape", "应答不是对象")
        return parsed


def _normalize_answers(answers: object) -> dict[str, object]:
    """SystemOne 的 answers 可能是 {name: value}，也可能是 [{name, choice}]。"""
    if isinstance(answers, dict):
        return {str(key): value for key, value in answers.items()}
    if isinstance(answers, list):
        out: dict[str, object] = {}
        for item in answers:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("id") or item.get("question")
            if not name:
                continue
            for key in ("choice", "score", "answer", "value", "result"):
                if key in item:
                    out[str(name)] = item[key]
                    break
        return out
    return {}


def _choice_to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        # SystemOne 的 answers[name] 是 {type, choice, probabilities} 这种结构
        return _choice_to_bool(value.get("choice") or value.get("answer") or value.get("value"))
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip()
        if token in {"收录", "是", "true", "yes", "y", "1"}:
            return True
        if token in {"不收录", "否", "false", "no", "n", "0"}:
            return False
    return False


def _score_to_unit(value: object) -> float:
    """把 0-1 / 1-5 / 0-100 三种常见刻度都归一化到 0-1。"""
    if isinstance(value, bool):
        return 0.0
    number: float | None = None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped in _SCORE_LEVELS:
            return _SCORE_LEVELS.index(stripped) / (len(_SCORE_LEVELS) - 1)
        try:
            number = float(stripped)
        except ValueError:
            return 0.0
    if number is None or number != number:
        return 0.0
    if number <= 1.0:
        return min(max(number, 0.0), 1.0)
    if number <= 5.0:
        return min(max((number - 1.0) / 4.0, 0.0), 1.0)
    return min(max(number / 100.0, 0.0), 1.0)
