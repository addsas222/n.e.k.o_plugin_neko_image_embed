"""两层：识图（perception）→ 判断（judgment）。

**识图**只回答「图里有什么」：Sensenova 云端视觉，失败退本地 CLIP 零样本。
**判断**只回答「这张图对这个角色值不值得记」，候选固定三种：

    model       —— 一个判断模型端点（SystemOne 或 OpenAI 兼容，laya/kev/jev 都只是它的实现）
    host        —— 宿主模型槽位（ctx.models + [plugin.models.judge]）
    heuristic   —— 纯启发式，永远可用，也是前两者的兜底

判断失败只降级、不伪造：错误逐级记进 ``errors``，最终一定落到 heuristic。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .config import Settings
from .judges.base import JudgeError, JudgeRequest, Verdict
from .judges.gateway import GatewayJudge
from .judges.http_judge import HttpJudge
from .judges.rules import RulesJudge
from .judges.zeroshot import ZeroShotJudge

JUDGE_MODES = ("model", "host", "heuristic")


class JudgeChainError(RuntimeError):
    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("所有判定引擎都失败")
        self.errors = errors


@dataclass(frozen=True)
class JudgeOutcome:
    verdict: Verdict
    errors: dict[str, str] = field(default_factory=dict)
    tried: tuple[str, ...] = ()
    perception: str = ""

    @property
    def degraded(self) -> bool:
        return bool(self.errors)


class JudgeChain:
    def __init__(
        self,
        settings: Settings,
        *,
        ctx: object | None = None,
        embedder: object | None = None,
        cache_dir: object | None = None,
    ) -> None:
        self._settings = settings
        self._ctx = ctx
        self._embedder = embedder
        self._cache_dir = cache_dir
        self._rules = RulesJudge(settings.limits)
        self._perception = self._build_perception()
        self._judges = self._build_judges()

    # ------------------------------------------------------------------ 装配

    def _build_perception(self) -> list[tuple[str, object]]:
        cfg = self._settings.judge
        out: list[tuple[str, object]] = []
        for name in cfg.perception:
            if name == "zeroshot":
                if self._embedder is not None:
                    out.append(
                        (
                            "zeroshot",
                            ZeroShotJudge(
                                self._embedder,
                                min_value=self._settings.limits.min_value,
                                cache_dir=self._cache_dir,
                            ),
                        )
                    )
                continue
            engine = cfg.engines.get(name)
            if engine is not None and engine.ready():
                out.append(
                    (
                        name,
                        HttpJudge(
                            engine,
                            timeout=cfg.timeout_seconds,
                            max_tokens=cfg.max_completion_tokens,
                        ),
                    )
                )
        return out

    def _build_judges(self) -> list[tuple[str, object | None]]:
        cfg = self._settings.judge
        mode = cfg.mode if cfg.mode in JUDGE_MODES else "heuristic"
        out: list[tuple[str, object | None]] = []
        if mode == "model":
            engine = cfg.engines.get("model")
            if engine is not None and engine.ready():
                out.append(
                    (
                        "model",
                        HttpJudge(
                            engine,
                            timeout=cfg.timeout_seconds,
                            max_tokens=cfg.max_completion_tokens,
                        ),
                    )
                )
        elif mode == "host":
            gateway_cfg = cfg.engines.get("gateway")
            if gateway_cfg is not None and gateway_cfg.enabled:
                out.append(
                    (
                        "host",
                        None
                        if self._ctx is None
                        else GatewayJudge(
                            self._ctx,
                            usage="judge",
                            timeout=cfg.timeout_seconds,
                            max_tokens=cfg.max_completion_tokens,
                        ),
                    )
                )
        return out

    @property
    def mode(self) -> str:
        cfg = self._settings.judge
        return cfg.mode if cfg.mode in JUDGE_MODES else "heuristic"

    @property
    def engine_names(self) -> list[str]:
        return [name for name, _ in self._judges]

    @property
    def perception_names(self) -> list[str]:
        return [name for name, _ in self._perception]

    # ------------------------------------------------------------------ 编排

    async def judge(self, request: JudgeRequest) -> JudgeOutcome:
        errors: dict[str, str] = {}
        tried: list[str] = []

        caption = ""
        tags: tuple[str, ...] = ()
        perception_name = ""
        for name, engine in self._perception:
            tried.append(f"perceive:{name}")
            try:
                perceived = await engine.judge(request)  # type: ignore[attr-defined]
            except JudgeError as exc:
                errors[name] = f"{exc.kind}: {exc.detail}"
                continue
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"unexpected: {type(exc).__name__}: {exc}"
                continue
            caption = (perceived.caption or "").strip()
            tags = tuple(perceived.tags or ())
            perception_name = name
            break

        judged = replace(
            request,
            caption_hint=caption or request.caption_hint,
            tags=tags or request.tags,
        )

        for name, judge in self._judges:
            tried.append(name)
            if judge is None:
                errors[name] = "ctx 不可用（插件上下文缺失）"
                continue
            try:
                verdict = await judge.judge(judged)  # type: ignore[attr-defined]
            except JudgeError as exc:
                errors[name] = f"{exc.kind}: {exc.detail}"
                continue
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"unexpected: {type(exc).__name__}: {exc}"
                continue
            return JudgeOutcome(
                verdict=self._merge(verdict, caption, tags, perception_name),
                errors=errors,
                tried=tuple(tried),
                perception=perception_name,
            )

        tried.append("heuristic")
        try:
            verdict = self._rules.judge(judged)
        except Exception as exc:  # noqa: BLE001
            errors["heuristic"] = f"{type(exc).__name__}: {exc}"
            raise JudgeChainError(errors) from exc
        return JudgeOutcome(
            verdict=self._merge(verdict, caption, tags, perception_name),
            errors=errors,
            tried=tuple(tried),
            perception=perception_name,
        )

    @staticmethod
    def _merge(
        verdict: Verdict, caption: str, tags: tuple[str, ...], perception: str
    ) -> Verdict:
        """判断层可能不产出 caption/tags（heuristic/model 只打分），用识图结果补齐。"""
        return replace(
            verdict,
            caption=verdict.caption or caption,
            tags=list(verdict.tags or tags),
            perception=perception,
        )
