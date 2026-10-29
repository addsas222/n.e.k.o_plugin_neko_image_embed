"""零样本主题判定：用 CLIP 自己的文本塔给图打中文标签。

为什么需要这一级：宿主模型网关没绑槽位时（本机默认状态），gateway 必 403，
链路直接掉到 rules——rules 只能看清晰度/熵，产不出 caption 和 tags，于是
（a）中文检索只能靠别名表硬猜，（b）`img_send` 发出去没有文字。
CLIP 的图文对齐本来就是现成的：拿一组主题提示词算余弦、softmax 取前几名，
就能在不引入任何新模型、不联网、不要密钥的前提下得到可用的中文标签。

它只做「图里大概是什么」，不判断「对角色有没有价值」——所以 ``issues`` 里会
明确标 ``persona_not_evaluated``，只在人设文本里出现同名主题时做一次轻加权。
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import io
from pathlib import Path

import numpy as np

from ..media import atomic_write
from .base import JudgeError, JudgeRequest, Verdict

# (中文标签, 英文提示词)。提示词按 CLIP 的训练分布写成短句，别加花。
TOPICS: tuple[tuple[str, str], ...] = (
    ("猫", "a photo of a cat"),
    ("狗", "a photo of a dog"),
    ("鸟", "a photo of a bird"),
    ("鱼", "a photo of a fish"),
    ("兔子", "a photo of a rabbit"),
    ("狮子", "a photo of a lion"),
    ("老虎", "a photo of a tiger"),
    ("马", "a photo of a horse"),
    ("昆虫", "a close-up photo of an insect"),
    ("人物", "a photo of a person"),
    ("自拍", "a selfie of a person"),
    ("风景", "a landscape photo"),
    ("天空", "a photo of the sky"),
    ("海", "a photo of the sea"),
    ("山", "a photo of a mountain"),
    ("花", "a photo of a flower"),
    ("建筑", "a photo of a building"),
    ("城市", "a city street scene"),
    ("食物", "a photo of food"),
    ("甜点", "a photo of a dessert"),
    ("饮料", "a photo of a drink"),
    ("表情包", "a meme or a reaction sticker"),
    ("截图", "a screenshot of a computer or phone screen"),
    ("动漫", "anime style artwork"),
    ("插画", "an illustration or a drawing"),
    ("文档", "a photo of a document with text"),
    ("室内", "an indoor scene"),
    ("夜景", "a night scene"),
    ("汽车", "a photo of a car"),
    ("游戏", "a video game screenshot"),
)

# 温度：CLIP 的图文余弦被压得很扁（常见 0.2–0.35），不放大就分不出高下。
TEMPERATURE = 0.012
# 一个标签要被采纳，概率至少是均分（1/24≈0.042）的 3 倍。
MIN_PROB = 0.12
MAX_TAGS = 3


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    total = float(exp.sum())
    return exp / total if total > 0 else np.zeros_like(exp)


class ZeroShotJudge:
    """本地语义级：CLIP 零样本打标 + 人设轻加权。"""

    engine = "zeroshot"
    modality = "local_semantic"

    def __init__(
        self,
        embedder: object,
        *,
        min_value: float = 0.45,
        cache_dir: object | None = None,
    ) -> None:
        self._embedder = embedder
        self._min_value = float(min_value)
        self._text_vectors: np.ndarray | None = None
        self._cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        identity = str(getattr(self._embedder, "id", "") or "").strip()
        if not identity:
            return None
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in identity)
        return self._cache_dir / f"topics-{safe[:80]}.npy"

    def _load_cache(self) -> np.ndarray | None:
        path = self._cache_path()
        if path is None or not path.is_file():
            return None
        try:
            matrix = np.load(io.BytesIO(path.read_bytes()))
        except (OSError, ValueError):
            return None
        if matrix.ndim != 2 or matrix.shape[0] != len(TOPICS) or matrix.shape[1] <= 0:
            return None
        return np.asarray(matrix, dtype=np.float32)

    def _save_cache(self, matrix: np.ndarray) -> None:
        path = self._cache_path()
        if path is None:
            return
        buffer = io.BytesIO()
        np.save(buffer, matrix)
        with contextlib.suppress(OSError):
            atomic_write(path, buffer.getvalue())

    async def _topic_matrix(self) -> np.ndarray:
        if self._text_vectors is not None:
            return self._text_vectors

        cached = self._load_cache()
        if cached is not None:
            self._text_vectors = cached
            return cached

        prompts = [prompt for _label, prompt in TOPICS]
        batch = getattr(self._embedder, "embed_texts", None)
        if callable(batch):
            matrix = np.asarray(await batch(prompts), dtype=np.float32)
        else:
            matrix = np.vstack(
                [np.asarray(await self._embedder.embed_text(p), dtype=np.float32) for p in prompts]
            )
        if matrix.shape[0] != len(TOPICS) or matrix.ndim != 2:
            raise JudgeError(self.engine, "matrix", f"主题矩阵形状异常: {matrix.shape}")

        self._save_cache(matrix)
        self._text_vectors = matrix
        return matrix

    async def judge(self, request: JudgeRequest) -> Verdict:
        if not request.image_b64:
            raise JudgeError(self.engine, "no_image", "零样本判定需要预览图")
        try:
            raw = base64.b64decode(request.image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise JudgeError(self.engine, "decode", f"预览图解码失败: {exc}") from exc

        try:
            image_vector, mode = await self._embedder.embed_image(raw)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - 编码器失败交给链降级
            raise JudgeError(self.engine, "embed", f"{type(exc).__name__}: {exc}") from exc
        if mode == "caption":
            # 视觉塔不可用、退化成文本 caption 时，零样本没有意义
            raise JudgeError(self.engine, "embed", "视觉塔不可用，跳过零样本")

        matrix = await self._topic_matrix()
        scores = matrix @ np.asarray(image_vector, dtype=np.float32)
        probs = softmax(scores / TEMPERATURE)

        order = np.argsort(-probs)
        tags = [
            TOPICS[int(index)][0]
            for index in order
            if float(probs[int(index)]) >= MIN_PROB
        ][:MAX_TAGS]
        top_prob = float(probs[int(order[0])]) if order.size else 0.0

        value = self._value(request, top_prob, tags)
        caption = f"可能是：{'、'.join(tags)}" if tags else ""

        return Verdict(
            should_embed=value >= self._min_value,
            value=value,
            risk={},
            caption=caption,
            tags=tags,
            sticker="表情包" in tags,
            reason=(
                f"本地零样本：{'/'.join(tags) or '未命中主题'}"
                f"（top {top_prob:.2f}）；未做人设价值判断"
            ),
            engine=self.engine,
            modality=self.modality,
            issues=("persona_not_evaluated",),
        )

    def _value(self, request: JudgeRequest, top_prob: float, tags: list[str]) -> float:
        """质量打底 + 主题置信度 + 人设同名主题的轻加权，上限 0.9。"""
        base = 0.35
        quality = 0.0
        metrics = request.metrics
        if metrics is not None:
            quality = min(metrics.sharpness / 300.0, 1.0) * 0.2
        topic = min(top_prob, 1.0) * 0.3

        persona = (request.persona or "").lower()
        hits = sum(1 for tag in tags if tag and tag.lower() in persona)
        persona_bonus = min(hits * 0.12, 0.24)

        return round(min(base + quality + topic + persona_bonus, 0.9), 4)
