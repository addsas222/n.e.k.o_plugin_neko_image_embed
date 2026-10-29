"""rules 级：不依赖任何模型与网络的兜底。

它能做的只有「质量够不够格进库」，做不了语义与人设判断。所以它的输出必须
**自我暴露**：``engine="rules"``、``modality="heuristic"``、reason 里写明没做语义判断。
把启发式结果伪装成语义判断，比直接失败更糟。
"""

from __future__ import annotations

from ..config import Limits
from .base import JudgeRequest, Verdict


class RulesJudge:
    engine = "rules"
    modality = "heuristic"

    def __init__(self, limits: Limits) -> None:
        self._limits = limits

    def judge(self, request: JudgeRequest) -> Verdict:
        metrics = request.metrics
        if metrics is None:
            return Verdict(
                should_embed=False,
                value=0.0,
                reason="没有图像指标，规则引擎无法判断",
                engine=self.engine,
                modality=self.modality,
            )

        quality = min(metrics.sharpness / (self._limits.min_sharpness * 3.0 or 1.0), 1.0)
        detail = min(metrics.unique_colors / 2000.0, 1.0)
        min_edge = min(metrics.width, metrics.height)
        size_fit = 1.0 if 200 <= min_edge <= 2048 else 0.5
        base = 0.45 * quality + 0.25 * detail + 0.30 * size_fit

        # 识图给出的标签/描述命中人设时，做一次轻加权——这是启发式唯一能碰到「人设」的地方。
        persona = (request.persona or "").lower()
        hits = [tag for tag in (request.tags or ()) if tag and tag.lower() in persona]
        if not hits and request.caption_hint:
            hits = [word for word in ("猫", "狗", "头像", "表情") if word in request.caption_hint and word in persona]
        persona_bonus = min(len(hits) * 0.12, 0.36)
        value = round(min(base + persona_bonus, 0.95), 4)

        sticker = min_edge <= 512 and 0.4 <= metrics.aspect <= 2.5
        return Verdict(
            should_embed=value >= self._limits.min_value,
            value=value,
            risk={},
            caption="",
            tags=[],
            sticker=sticker,
            reason=(
                f"仅质量启发式（清晰度 {metrics.sharpness}、主色数 {metrics.unique_colors}、"
                f"短边 {min_edge}）"
                + (f"；人设命中 {('/'.join(hits))} +{persona_bonus:.2f}" if hits else "；未命中人设关键词")
                + "；未做语义判断"
            ),
            engine=self.engine,
            modality=self.modality,
            issues=("semantic_judgement_skipped",),
        )
