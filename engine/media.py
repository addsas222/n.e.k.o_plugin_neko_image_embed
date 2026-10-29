"""文件与哈希工具：原子写、路径越界防护、dHash。

这里刻意只依赖标准库 + Pillow，不碰 sqlite，也不 import SDK，
这样 ``img_gc`` 的路径安全逻辑可以单独测。
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from pathlib import Path

from PIL import Image

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """git 对象 id：``sha1("blob <len>\\0" + content)``。

    HF 上的小文件不是 LFS，只给 blobId；用这个算法就能在不依赖 git 的前提下校验。
    大文件走 LFS 的 sha256，见 models.lock.json。
    """
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def atomic_write(target: Path, data: bytes) -> None:
    """同目录临时文件 + os.replace，避免半截文件被读到。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - 需要先拿到名字再 os.replace
        mode="wb", dir=str(target.parent), prefix=target.name + ".", suffix=".part", delete=False
    )
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise


def safe_child(base: Path, name: str) -> Path:
    """拼一个 base 下的路径，拒绝 ``..``、绝对路径、盘符与符号链接逃逸。

    调用方约定：base 必须是已经 resolve 过的目录。
    """
    if not name or "\x00" in name:
        raise ValueError("文件名不能为空")
    candidate = Path(name)
    if candidate.is_absolute() or candidate.drive or len(candidate.parts) != 1:
        raise ValueError(f"非法文件名: {name!r}")
    resolved_base = base.resolve()
    resolved = (resolved_base / candidate).resolve()
    if resolved.parent != resolved_base:
        raise ValueError(f"路径越界: {name!r}")
    return resolved


def safe_relative(base: Path, rel: str) -> Path:
    """``rel`` 允许子目录（如 ``media/<sha>.jpg``），但必须落在 ``base`` 内。"""
    if not rel or "\x00" in rel:
        raise ValueError("相对路径不能为空")
    candidate = Path(rel)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError(f"非法相对路径: {rel!r}")
    resolved_base = base.resolve()
    resolved = (resolved_base / candidate).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise ValueError(f"路径越界: {rel!r}")
    return resolved


def hamming(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """差值哈希：对相邻像素做差，得到 ``hash_size ** 2`` 位整数。"""
    import numpy as np

    small = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(small, dtype=np.int16)
    bits = 0
    for value in (pixels[:, 1:] > pixels[:, :-1]).ravel():
        bits = (bits << 1) | int(value)
    return bits


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{int(size)} B"


_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def sniff_mime(data: bytes) -> str:
    """按魔数嗅探；认不出就交给 Pillow 判 format，再不行当 octet-stream。"""
    head = data[:16]
    for magic, mime in _MAGIC:
        if head.startswith(magic):
            return mime
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"
