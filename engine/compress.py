"""阶段 3：压缩。

三个硬目标，缺一不可：

1. **去元数据**：EXIF/GPS/ICC 一律不落盘。做法是从像素重建 Image，而不是 copy()。
2. **体积可控**：``store_max_bytes`` 是 ``push_message`` 单条 512 KiB 上限的安全线，
   压缩必须真的把结果压到这个预算内，压不下去就报错而不是悄悄超发。
3. **可预期**：只缩不放；动图在 guard 阶段已被拒或标 flattened，这里只处理静帧。
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

_QUALITY_FLOOR = 48
_MIN_EDGE = 320


class CompressError(RuntimeError):
    """在给定预算内压不下去。"""


@dataclass(frozen=True)
class Compressed:
    data: bytes
    width: int
    height: int
    mime: str
    quality: int
    bytes: int


def strip_metadata(image: Image.Image) -> Image.Image:
    """从像素重建，彻底丢掉 info/exif/icc。"""
    rgb = image.convert("RGB")
    return Image.frombytes("RGB", rgb.size, rgb.tobytes())


def fit_within(image: Image.Image, long_edge: int) -> Image.Image:
    if max(image.size) <= long_edge:
        return image
    copy = image.copy()
    copy.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    return copy


def compress(
    image: Image.Image,
    *,
    long_edge: int,
    quality: int,
    max_bytes: int,
) -> Compressed:
    """先降质量、再降尺寸，直到落进预算；仍不行则抛 CompressError。"""
    base = fit_within(strip_metadata(image), long_edge)
    current_quality = quality

    for _ in range(10):
        buffer = io.BytesIO()
        base.save(buffer, format="JPEG", quality=current_quality, optimize=True, progressive=True)
        payload = buffer.getvalue()
        if len(payload) <= max_bytes:
            return Compressed(
                data=payload,
                width=base.width,
                height=base.height,
                mime="image/jpeg",
                quality=current_quality,
                bytes=len(payload),
            )
        if current_quality > _QUALITY_FLOOR:
            current_quality = max(_QUALITY_FLOOR, current_quality - 8)
            continue
        if max(base.size) <= _MIN_EDGE:
            break
        smaller = base.copy()
        smaller.thumbnail(
            (int(base.width * 0.8) or 1, int(base.height * 0.8) or 1),
            Image.Resampling.LANCZOS,
        )
        base = smaller
        current_quality = quality

    raise CompressError(
        f"压不进 {max_bytes} 字节（最后 {base.width}x{base.height} q{current_quality}）"
    )


def make_thumb(image: Image.Image, *, edge: int, quality: int) -> bytes:
    """缩略图允许失败：调用方把空 bytes 当成「没有缩略图」即可。"""
    thumb = strip_metadata(image)
    thumb.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()
