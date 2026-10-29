"""判定链测试：降级不伪造、JSON 容错、字段兜底。"""

from __future__ import annotations

import asyncio

import numpy as np
from neko_image_embed.engine.config import load_settings
from neko_image_embed.engine.guard import measure
from neko_image_embed.engine.judge import JudgeChain
from neko_image_embed.engine.judges.base import (
    JudgeRequest,
    coerce_verdict,
    parse_json_loose,
)
from neko_image_embed.engine.judges.http_judge import (
    _join,
    _normalize_answers,
    _score_to_unit,
)
from PIL import Image


def _request() -> JudgeRequest:
    rng = np.random.default_rng(2)
    image = Image.fromarray(rng.integers(0, 256, (200, 200, 3), dtype=np.uint8), "RGB")
    return JudgeRequest(persona="角色：YUI", digest="尺寸=200x200", metrics=measure(image))


def test_chain_degrades_to_rules_and_keeps_errors() -> None:
    settings = load_settings({"image_embed": {"judge": {"order": ["gateway", "rules"]}}})
    chain = JudgeChain(settings, ctx=None)
    assert chain.engine_names == ["host"]

    outcome = asyncio.run(chain.judge(_request()))
    assert outcome.verdict.engine == "rules"
    assert outcome.verdict.modality == "heuristic"
    assert "host" in outcome.errors
    assert "semantic_judgement_skipped" in outcome.verdict.issues
    assert outcome.degraded is True


def test_rules_can_accept_a_detailed_image() -> None:
    settings = load_settings({})
    chain = JudgeChain(settings, ctx=None)
    outcome = asyncio.run(chain.judge(_request()))
    assert 0.0 <= outcome.verdict.value <= 1.0


def test_parse_json_loose_handles_fences_and_noise() -> None:
    payload = parse_json_loose(
        '好的，结果如下：\n```json\n{"should_embed": true, "value": 0.8, "tags": ["猫"]}\n```\n'
    )
    assert payload["should_embed"] is True
    assert payload["value"] == 0.8


def test_parse_json_loose_handles_braces_inside_strings() -> None:
    payload = parse_json_loose('说明 {不是对象} 然后 {"reason": "包含 } 花括号", "value": 0.3}')
    assert payload["reason"] == "包含 } 花括号"
    assert payload["value"] == 0.3


def test_parse_json_loose_rejects_garbage() -> None:
    try:
        parse_json_loose("完全不是 JSON")
    except ValueError as exc:
        assert "JSON" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛 ValueError")


def test_coerce_verdict_fills_defaults_and_clamps() -> None:
    verdict = coerce_verdict(
        {"value": 3.5, "risk": {"nsfw": 2, "unknown": 1}, "tags": "猫, 猫, 日常"},
        engine="gateway",
        modality="vision",
    )
    assert verdict.should_embed is False
    assert verdict.value == 1.0
    assert verdict.risk == {"nsfw": 1.0}
    assert verdict.tags == ["猫", "日常"]
    assert any("should_embed" in issue for issue in verdict.issues)


class _TopicEmbedder:
    """把每个提示词映射到一根独立的轴，图像向量指向指定主题。"""

    def __init__(self, target_label: str) -> None:
        from neko_image_embed.engine.judges.zeroshot import TOPICS

        self.dim = len(TOPICS) + 2
        self._axis = {prompt: index for index, (_label, prompt) in enumerate(TOPICS)}
        self._target = next(prompt for label, prompt in TOPICS if label == target_label)

    async def embed_text(self, text: str):
        import numpy as np

        vector = np.zeros(self.dim, dtype=np.float32)
        vector[self._axis.get(text, self.dim - 1)] = 1.0
        return vector

    async def embed_image(self, data: bytes, *, caption: str = ""):
        import numpy as np

        vector = np.zeros(self.dim, dtype=np.float32)
        vector[self._axis[self._target]] = 1.0
        return vector, "vision"

    async def embed_texts(self, texts: list[str]):
        import numpy as np

        self.batch_calls = getattr(self, "batch_calls", 0) + 1
        return np.vstack([await self.embed_text(text) for text in texts])


def test_zeroshot_tags_and_caption() -> None:
    import base64

    from neko_image_embed.engine.judges.zeroshot import ZeroShotJudge

    judge = ZeroShotJudge(_TopicEmbedder("猫"), min_value=0.45)
    request = JudgeRequest(
        persona="角色：YUI，喜欢猫",
        digest="",
        image_b64=base64.b64encode(b"fake-jpeg").decode("ascii"),
    )
    verdict = asyncio.run(judge.judge(request))

    assert verdict.engine == "zeroshot"
    assert verdict.modality == "local_semantic"
    assert verdict.tags[0] == "猫"
    assert "猫" in verdict.caption
    assert verdict.should_embed is True
    assert "persona_not_evaluated" in verdict.issues


def test_zeroshot_persona_boost() -> None:
    import base64

    from neko_image_embed.engine.judges.zeroshot import ZeroShotJudge

    payload = base64.b64encode(b"fake-jpeg").decode("ascii")
    judge = ZeroShotJudge(_TopicEmbedder("猫"), min_value=0.45)
    plain = asyncio.run(judge.judge(JudgeRequest(persona="角色：YUI", digest="", image_b64=payload)))
    liked = asyncio.run(
        judge.judge(JudgeRequest(persona="角色：YUI，最喜欢猫", digest="", image_b64=payload))
    )
    assert liked.value > plain.value


def test_perception_feeds_the_judge() -> None:
    """识图（zeroshot）负责标签，判断层（默认启发式）负责价值；两者分开记。"""
    import base64

    settings = load_settings({})
    chain = JudgeChain(settings, ctx=None, embedder=_TopicEmbedder("猫"))
    assert chain.perception_names == ["zeroshot"]
    assert chain.engine_names == [], "默认判断层是启发式，没有外部引擎"

    request = JudgeRequest(
        persona="角色：YUI",
        digest="",
        image_b64=base64.b64encode(b"fake-jpeg").decode("ascii"),
    )
    outcome = asyncio.run(chain.judge(request))
    assert outcome.perception == "zeroshot"
    assert outcome.verdict.engine == "rules"
    assert outcome.verdict.perception == "zeroshot"
    assert "猫" in outcome.verdict.tags


def test_zeroshot_matrix_is_cached_on_disk(tmp_path) -> None:
    """主题矩阵只该编码一次；第二次（哪怕换了实例）要从缓存读。"""
    import base64

    from neko_image_embed.engine.judges.zeroshot import ZeroShotJudge

    payload = base64.b64encode(b"fake-jpeg").decode("ascii")
    request = JudgeRequest(persona="角色：YUI", digest="", image_b64=payload)

    embedder = _TopicEmbedder("狗")
    embedder.id = "fake-embedder"  # type: ignore[attr-defined]
    first = ZeroShotJudge(embedder, min_value=0.45, cache_dir=tmp_path)
    asyncio.run(first.judge(request))
    assert getattr(embedder, "batch_calls", 0) == 1
    assert list(tmp_path.glob("topics-*.npy")), "首次应写出缓存"

    reuse = _TopicEmbedder("狗")
    reuse.id = "fake-embedder"  # type: ignore[attr-defined]
    second = ZeroShotJudge(reuse, min_value=0.45, cache_dir=tmp_path)
    verdict = asyncio.run(second.judge(request))
    assert getattr(reuse, "batch_calls", 0) == 0, "命中缓存时不该再编码"
    assert verdict.tags[0] == "狗"


def test_http_helpers() -> None:
    assert _join("http://127.0.0.1:8009", "systemone") == "http://127.0.0.1:8009/v1/systemone"
    assert _join("http://x/v1", "chat/completions") == "http://x/v1/chat/completions"
    assert _score_to_unit("很高") == 1.0
    assert _score_to_unit(3) == 0.5
    assert _score_to_unit(80) == 0.8
    assert _normalize_answers([{"name": "should_embed", "choice": "收录"}]) == {
        "should_embed": "收录"
    }


class _BindingError(Exception):
    status_code = 403

    def __str__(self) -> str:
        return "Error code: 403 - model_usage_not_bound"


class _FakeCompletions:
    async def create(self, **_kwargs):
        raise _BindingError()


class _FakeClient:
    class chat:  # noqa: N801 - 只是给 duck typing 用的壳
        completions = _FakeCompletions()


class _FakeModels:
    def __init__(self) -> None:
        self.calls = 0

    async def get_client(self):
        self.calls += 1
        return _FakeClient()


class _FakeCtx:
    def __init__(self) -> None:
        self.models = _FakeModels()


def test_binding_error_is_detected() -> None:
    from neko_image_embed.engine.judges.gateway import _is_binding_error

    assert _is_binding_error(_BindingError()) is True
    assert _is_binding_error(RuntimeError("model_usage_not_bound")) is False  # 没有 403


def test_gateway_cools_down_after_unbound(monkeypatch) -> None:
    """未绑定时第二次调用必须立刻失败，不能每张图都等一次 403。"""
    from neko_image_embed.engine.judges import gateway as gateway_mod
    from neko_image_embed.engine.judges.base import JudgeError

    # 真实 openai 包冷导入约 5 秒；单测里没必要为它付这笔钱。
    monkeypatch.setattr(gateway_mod, "ensure_sdk_imported", lambda: True)

    ctx = _FakeCtx()
    judge = gateway_mod.GatewayJudge(ctx, usage="judge", timeout=5.0, max_tokens=128)

    async def scenario() -> tuple[str, str]:
        first_kind = ""
        try:
            await judge.judge(_request())
        except JudgeError as exc:
            first_kind = exc.kind
        second_kind = ""
        try:
            await judge.judge(_request())
        except JudgeError as exc:
            second_kind = exc.kind
        return first_kind, second_kind

    first, second = asyncio.run(scenario())
    assert first == "call"
    assert second == "unbound"
    assert ctx.models.calls == 1
