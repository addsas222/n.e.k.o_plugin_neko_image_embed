"""模型自举：按 models.lock.json 下载并校验 CLIP 权重。

安全约定：
* 大文件走 LFS 的 sha256，小文件走 git blob sha1（HF 只给 blobId），两者都能离线复核；
* 先写 ``.part`` 再原子替换，半截文件不会被当成可用模型；
* ``allow_download=false`` 时只做存在性检查，绝不偷偷联网。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..media import atomic_write, git_blob_sha1, sha256_bytes

_CHUNK = 1 << 20


@dataclass(frozen=True)
class ModelState:
    ok: bool
    detail: str
    downloaded: bool = False


def load_lock(lock_path: Path) -> dict:
    try:
        raw = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def verify_file(path: Path, algo: str, digest: str) -> bool:
    if not digest:
        return path.is_file()
    try:
        data = Path(path).read_bytes()
    except OSError:
        return False
    if algo == "sha256":
        return sha256_bytes(data) == digest
    if algo == "git-sha1":
        return git_blob_sha1(data) == digest
    return False


def ensure_model(
    model_dir: Path,
    *,
    lock_path: Path,
    endpoint: str,
    allow_download: bool,
    timeout: float = 120.0,
    progress: object | None = None,
) -> ModelState:
    """保证模型目录可用。返回的状态里 detail 一定是人能看懂的原因。"""
    model_dir = Path(model_dir)
    lock = load_lock(lock_path)
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        return ModelState(False, f"模型清单不可用: {lock_path}")

    repo = str(lock.get("repo") or "")
    revision = str(lock.get("revision") or "main")
    base = f"{endpoint.rstrip('/')}/{repo}/resolve/{revision}"

    downloaded = False
    for entry in files:
        if not isinstance(entry, dict):
            continue
        local = str(entry.get("local") or "")
        if not local:
            continue
        target = model_dir / local
        algo = str(entry.get("algo") or "")
        digest = str(entry.get("digest") or "")
        if target.is_file() and verify_file(target, algo, digest):
            continue
        if not allow_download:
            return ModelState(False, f"缺少 {local} 且 allow_download=false")
        remote = str(entry.get("remote") or local)
        url = f"{base}/{remote}"
        try:
            payload = _fetch(url, timeout=timeout, progress=progress, name=local)
        except RuntimeError as exc:
            return ModelState(False, f"下载 {local} 失败: {exc}", downloaded)
        if not verify_file_bytes(payload, algo, digest):
            return ModelState(False, f"{local} 校验失败（{algo} 不匹配）", downloaded)
        try:
            atomic_write(target, payload)
        except OSError as exc:
            return ModelState(False, f"写入 {local} 失败: {exc}", downloaded)
        downloaded = True

    return ModelState(True, str(model_dir), downloaded)


def verify_file_bytes(data: bytes, algo: str, digest: str) -> bool:
    if not digest:
        return True
    if algo == "sha256":
        return sha256_bytes(data) == digest
    if algo == "git-sha1":
        return git_blob_sha1(data) == digest
    return False


def _fetch(url: str, *, timeout: float, progress: object | None, name: str) -> bytes:
    request = urllib.request.Request(url, headers={"user-agent": "neko-image-embed/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                block = response.read(_CHUNK)
                if not block:
                    break
                chunks.append(block)
                total += len(block)
                if callable(progress):
                    progress(name, total)
            return b"".join(chunks)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"网络错误 {exc}") from exc
