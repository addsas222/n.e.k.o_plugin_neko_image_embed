"""压缩测试：预算、去元数据、只缩不放。"""

from __future__ import annotations

import io

import numpy as np
import pytest
from neko_image_embed.engine.compress import CompressError, compress, make_thumb
from PIL import Image


def _noise(width: int = 3000, height: int = 2000, seed: int = 11) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, "RGB")


def test_result_fits_budget_and_is_jpeg() -> None:
    result = compress(_noise(), long_edge=1280, quality=82, max_bytes=512 * 1024)
    assert result.bytes <= 512 * 1024
    assert result.data[:2] == b"\xff\xd8"
    assert max(result.width, result.height) == 1280


def test_small_image_is_not_upscaled() -> None:
    small = _noise(200, 120)
    result = compress(small, long_edge=1280, quality=82, max_bytes=512 * 1024)
    assert (result.width, result.height) == (200, 120)


def test_metadata_is_stripped() -> None:
    image = _noise(640, 480)
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation
    exif[0x010F] = "TestCam"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    with Image.open(io.BytesIO(buffer.getvalue())) as loaded:
        assert loaded.getexif()
        loaded.load()
        result = compress(loaded, long_edge=1280, quality=82, max_bytes=512 * 1024)

    with Image.open(io.BytesIO(result.data)) as out:
        assert not dict(out.getexif())


def test_impossible_budget_raises() -> None:
    with pytest.raises(CompressError):
        compress(_noise(800, 600), long_edge=1280, quality=82, max_bytes=64)


def test_thumbnail_respects_edge() -> None:
    payload = make_thumb(_noise(1600, 900), edge=256, quality=75)
    with Image.open(io.BytesIO(payload)) as thumb:
        assert max(thumb.size) <= 256
        assert thumb.format == "JPEG"
