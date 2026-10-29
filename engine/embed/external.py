"""外部 embeddings 后端（OpenAI 兼容 ``/embeddings``）。

它的输入只能是文本，所以图像走这里时用的是判定链给出的 caption。这意味着
**向量空间与 CLIP 不同**：切换后端必须重建索引，store 的空间守卫会拦住混用。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np

from ..config import EmbedderConfig


class ExternalError(RuntimeError):
    pass


class ExternalEmbedder:
    def __init__(self, cfg: EmbedderConfig, *, timeout: float = 30.0) -> None:
        self._cfg = cfg
        self._timeout = max(1.0, timeout)
        self.dim = int(cfg.external_dim or 0)
        self.id = f"external:{cfg.external_model or 'unknown'}"

    def configured(self) -> tuple[bool, str]:
        if not self._cfg.external_base_url:
            return False, "external_base_url 未配置"
        if not self._cfg.external_model:
            return False, "external_model 未配置"
        return True, ""

    def embed(self, text: str) -> np.ndarray:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ExternalError("外部 embeddings 只接受文本，且文本不能为空")
        ok, detail = self.configured()
        if not ok:
            raise ExternalError(detail)

        base = self._cfg.external_base_url.rstrip("/")
        url = base + ("/embeddings" if base.endswith("/v1") else "/v1/embeddings")
        body = json.dumps(
            {"model": self._cfg.external_model, "input": cleaned[:4000]},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self._cfg.external_api_key:
            headers["authorization"] = f"Bearer {self._cfg.external_api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise ExternalError(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ExternalError(f"连接失败: {exc}") from exc

        try:
            parsed = json.loads(raw.decode("utf-8"))
            vector = np.asarray(parsed["data"][0]["embedding"], dtype=np.float32)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ExternalError(f"应答结构异常: {exc}") from exc
        if vector.size == 0:
            raise ExternalError("空向量")
        if self.dim and vector.size != self.dim:
            raise ExternalError(f"维度 {vector.size} 与配置 external_dim={self.dim} 不一致")
        self.dim = int(vector.size)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else vector
