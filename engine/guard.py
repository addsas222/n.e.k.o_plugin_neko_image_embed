"""阶段 1：合规与质量闸门。

这一层**不调用任何模型**，职责是把「明显不该进库」的输入挡在外面，并且给出足够的
指标供后面两级使用：

* 结构合规：字节数、像素数（防解压炸弹）、能否解码、动图策略；
* 隐私合规：EXIF（含 GPS）在这里随解码一起丢弃，绝不落到磁盘；
* 价值粗筛：清晰度、熵、亮度/对比度、纯色、近似重复（dHash）。

真正的语义价值判断交给 judge 链；这里只做「不合格就拒」和「指标采集」。
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Limits
from .media import dhash, sha256_bytes


@dataclass(frozen=True)
class Probe:
    format: str
    width: int
    height: int
    mode: str
    frames: int
    has_alpha: bool


@dataclass(frozen=True)
class Metrics:
    width: int
    height: int
    aspect: float
    sharpness: float
    brightness: float
    contrast: float
    entropy: float
    unique_colors: int
    dhash: int
    pixel_sha: str


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    code: str
    detail: str
    probe: Probe | None = None
    metrics: Metrics | None = None
    image: Image.Image | None = None
    flattened: bool = False


def inspect_image(data: bytes, limits: Limits, *, allow_animated: bool = False) -> GuardResult:
    """解码 + 体检。任何异常都收敛成 ``GuardResult(ok=False)``，不向上抛。"""
    if not data:
        return GuardResult(False, "empty", "空字节")
    if len(data) > limits.max_source_bytes:
        return GuardResult(
            False,
            "too_large",
            f"{len(data)} 字节超过上限 {limits.max_source_bytes}",
        )

    try:
        with Image.open(BytesIO(data)) as head:
            head.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return GuardResult(False, "not_image", f"{type(exc).__name__}: {exc}")

    try:
        with Image.open(BytesIO(data)) as raw:
            frames = int(getattr(raw, "n_frames", 1) or 1)
            fmt = str(raw.format or "")
            mode = raw.mode
            width, height = raw.size
            has_alpha = mode in {"RGBA", "LA", "PA"} or "transparency" in raw.info
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return GuardResult(False, "not_image", f"{type(exc).__name__}: {exc}")

    probe = Probe(fmt, width, height, mode, frames, has_alpha)

    if width < 16 or height < 16:
        return GuardResult(False, "too_small", f"{width}x{height} 太小", probe)
    if width * height > limits.max_source_pixels:
        return GuardResult(
            False,
            "too_many_pixels",
            f"{width * height} 像素超过上限 {limits.max_source_pixels}",
            probe,
        )
    if frames > 1 and not allow_animated:
        return GuardResult(
            False,
            "animated_not_allowed",
            f"动图 {frames} 帧；allow_animated=false 时只接受静图（避免静默丢帧）",
            probe,
        )

    try:
        with Image.open(BytesIO(data)) as raw:
            raw.seek(0)
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except (OSError, ValueError) as exc:
        return GuardResult(False, "decode_failed", f"{type(exc).__name__}: {exc}", probe)

    metrics = measure(image)
    if metrics.entropy < limits.min_entropy:
        return GuardResult(
            False,
            "low_entropy",
            f"灰度熵 {metrics.entropy:.2f} < {limits.min_entropy}（近乎纯色）",
            probe,
            metrics,
            image,
        )
    if metrics.sharpness < limits.min_sharpness:
        return GuardResult(
            False,
            "blurry",
            f"清晰度 {metrics.sharpness:.1f} < {limits.min_sharpness}",
            probe,
            metrics,
            image,
        )

    return GuardResult(True, "ok", "", probe, metrics, image, frames > 1)


def measure(image: Image.Image) -> Metrics:
    """在不超过 256px 的副本上算指标，避免大图把 CPU 拖住。"""
    width, height = image.size
    small = image.copy()
    small.thumbnail((256, 256), Image.Resampling.LANCZOS)

    gray = np.asarray(small.convert("L"), dtype=np.float32)
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    inner = lap[1:-1, 1:-1]
    sharpness = float(inner.var()) if inner.size else 0.0

    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = float(hist.sum()) or 1.0
    prob = hist[hist > 0] / total
    entropy = float(-(prob * np.log2(prob)).sum())

    rgb_small = np.asarray(small.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS))
    unique_colors = int(len(np.unique(rgb_small.reshape(-1, 3), axis=0)))

    pixel_small = image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
    return Metrics(
        width=width,
        height=height,
        aspect=round(width / height, 4) if height else 0.0,
        sharpness=round(sharpness, 2),
        brightness=round(float(gray.mean()), 2),
        contrast=round(float(gray.std()), 2),
        entropy=round(entropy, 3),
        unique_colors=unique_colors,
        dhash=dhash(image),
        pixel_sha=sha256_bytes(pixel_small.tobytes()),
    )


def digest_for_prompt(metrics: Metrics | None, probe: Probe | None) -> str:
    """给纯文本引擎看的图像摘要（SystemOne 那类拿不到像素的端点）。"""
    if metrics is None or probe is None:
        return "（无图像指标）"
    return (
        f"格式={probe.format} 尺寸={metrics.width}x{metrics.height} "
        f"清晰度={metrics.sharpness} 亮度={metrics.brightness} 对比度={metrics.contrast} "
        f"熵={metrics.entropy} 主色数={metrics.unique_colors}"
    )
