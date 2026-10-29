"""CLIP ViT-B/32（ONNX，单文件全模型）。

为什么用 ``model_quantized.onnx`` 而不是两个独立塔：独立塔输出的是 768 维
hidden state/pooler，**文本与图像不在同一空间**，拿来做跨模态检索会得到
看似合理但毫无意义的相似度。全模型导出的 ``image_embeds`` / ``text_embeds``
才是共享空间（512 维）。

推理全部同步，由上层 ``asyncio.to_thread`` 包起来。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_SIZE = 224
MAX_TOKENS = 77


class ClipError(RuntimeError):
    pass


class ClipEmbedder:
    """模型目录里必须有 ``model.onnx`` 与 ``tokenizer.json``。"""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()
        self._output_names: dict[str, str] = {}
        self._mean = np.asarray(DEFAULT_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(DEFAULT_STD, dtype=np.float32).reshape(3, 1, 1)
        self.size = DEFAULT_SIZE
        self.dim = 0
        self.id = ""

    # ------------------------------------------------------------------ 加载

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.onnx"

    @property
    def tokenizer_path(self) -> Path:
        return self.model_dir / "tokenizer.json"

    def available(self) -> tuple[bool, str]:
        missing = [p.name for p in (self.model_path, self.tokenizer_path) if not p.is_file()]
        if missing:
            return False, f"模型文件缺失: {', '.join(missing)}"
        return True, str(self.model_dir)

    def load(self) -> None:
        if self._session is not None:
            return
        ok, detail = self.available()
        if not ok:
            raise ClipError(detail)

        try:
            import onnxruntime as ort
        except ImportError as exc:  # 宿主理应自带；缺了就是环境坏了
            raise ClipError(f"onnxruntime 不可用: {exc}") from exc

        try:
            session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - ORT 的异常类型很杂
            raise ClipError(f"加载模型失败: {type(exc).__name__}: {exc}") from exc

        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        except Exception as exc:  # noqa: BLE001
            raise ClipError(f"加载分词器失败: {type(exc).__name__}: {exc}") from exc

        self._session = session
        self._tokenizer = tokenizer
        self._input_names = {item.name for item in session.get_inputs()}
        self._output_names = self._resolve_outputs(session)
        self._read_preprocessor()
        self.id = f"clip-vit-base-patch32-{_short_hash(self.model_path)}"
        self.dim = self._probe_dim()

    def _resolve_outputs(self, session: object) -> dict[str, str]:
        names = [item.name for item in session.get_outputs()]  # type: ignore[attr-defined]
        lowered = {name.lower(): name for name in names}

        def pick(*candidates: str) -> str:
            for candidate in candidates:
                if candidate in lowered:
                    return lowered[candidate]
            return ""

        image = pick("image_embeds", "image_features", "visual_projection", "image_embedding")
        text = pick("text_embeds", "text_features", "text_projection", "text_embedding")
        shared = pick("embeds", "embedding")
        if not image:
            image = shared or pick("pooler_output", "last_hidden_state")
        if not text:
            text = shared or pick("pooler_output", "last_hidden_state")
        if not image or not text:
            raise ClipError(f"模型输出里找不到嵌入张量，实际输出: {names}")
        return {"image": image, "text": text}

    def _read_preprocessor(self) -> None:
        path = self.model_dir / "preprocessor_config.json"
        if not path.is_file():
            return
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(cfg, dict):
            return
        mean = cfg.get("image_mean")
        std = cfg.get("image_std")
        size = cfg.get("crop_size") or cfg.get("size")
        if isinstance(mean, list) and len(mean) == 3:
            self._mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        if isinstance(std, list) and len(std) == 3:
            self._std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        if isinstance(size, int) and size > 0:
            self.size = size
        elif isinstance(size, dict) and isinstance(size.get("height"), int):
            self.size = int(size["height"])

    def _probe_dim(self) -> int:
        """跑一张灰图确定维度并完成预热；失败就让上层降级。"""
        probe = Image.new("RGB", (self.size, self.size), (127, 127, 127))
        vector = self._run_image(probe)
        if vector.size <= 0:
            raise ClipError("探测向量维度失败")
        return int(vector.size)

    # ------------------------------------------------------------------ 推理

    def embed_image(self, image: Image.Image) -> np.ndarray:
        self.load()
        return self._run_image(image)

    def embed_text(self, text: str) -> np.ndarray:
        self.load()
        cleaned = (text or "").strip()
        if not cleaned:
            raise ClipError("空文本无法编码")
        return self._run_text(cleaned)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """批量文本编码，返回 (N, dim) 已归一化矩阵。

        零样本打标要一次编码几十条提示词，逐条跑要 3.5 秒；批量一次前向约 3 秒，
        再叠一层磁盘缓存（见 zeroshot），实际只在首次安装时付一次。
        """
        self.load()
        cleaned = [(text or "").strip() or " " for text in texts]
        if not cleaned:
            raise ClipError("空文本列表")
        if len(cleaned) == 1:
            return self._run_text(cleaned[0])[None, :]

        ids_list = []
        mask_list = []
        for text in cleaned:
            ids, mask = self._encode(text)
            ids_list.append(ids[0])
            mask_list.append(mask[0])
        feeds: dict[str, np.ndarray] = {
            "input_ids": np.stack(ids_list),
            "pixel_values": np.zeros(
                (len(cleaned), 3, self.size, self.size), dtype=np.float32
            ),
        }
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = np.stack(mask_list)
        try:
            outputs = self._session.run([self._output_names["text"]], feeds)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - 批量不支持就退回逐条
            return np.vstack([self._run_text(text) for text in cleaned])

        matrix = np.asarray(outputs[0], dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _run_image(self, image: Image.Image) -> np.ndarray:
        tensor = self._preprocess(image)
        feeds: dict[str, np.ndarray] = {}
        if "pixel_values" in self._input_names:
            feeds["pixel_values"] = tensor
        else:
            raise ClipError(f"模型缺少 pixel_values 输入，实际: {sorted(self._input_names)}")
        if "input_ids" in self._input_names:
            ids, mask = self._encode(" ")
            feeds["input_ids"] = ids
            if "attention_mask" in self._input_names:
                feeds["attention_mask"] = mask
        return self._infer(feeds, self._output_names["image"])

    def _run_text(self, text: str) -> np.ndarray:
        if "input_ids" not in self._input_names:
            raise ClipError(f"模型缺少 input_ids 输入，实际: {sorted(self._input_names)}")
        ids, mask = self._encode(text)
        feeds: dict[str, np.ndarray] = {"input_ids": ids}
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = mask
        if "pixel_values" in self._input_names:
            feeds["pixel_values"] = np.zeros(
                (1, 3, self.size, self.size), dtype=np.float32
            )
        return self._infer(feeds, self._output_names["text"])

    def _infer(self, feeds: dict[str, np.ndarray], output_name: str) -> np.ndarray:
        try:
            outputs = self._session.run([output_name], feeds)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            raise ClipError(f"推理失败: {type(exc).__name__}: {exc}") from exc
        array = np.asarray(outputs[0], dtype=np.float32)
        if array.ndim > 1:
            array = array.reshape(-1, array.shape[-1])[0]
        vector = array.reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise ClipError("编码结果为零向量")
        return vector / norm

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = self.size / min(width, height)
        new_size = (max(self.size, int(round(width * scale))), max(self.size, int(round(height * scale))))
        resized = rgb.resize(new_size, Image.Resampling.BICUBIC)
        left = (resized.width - self.size) // 2
        top = (resized.height - self.size) // 2
        cropped = resized.crop((left, top, left + self.size, top + self.size))
        array = np.asarray(cropped, dtype=np.float32) / 255.0
        array = array.transpose(2, 0, 1)
        array = (array - self._mean) / self._std
        return array[None, ...].astype(np.float32)

    def _encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        encoding = self._tokenizer.encode(text)  # type: ignore[union-attr]
        ids = list(encoding.ids)[:MAX_TOKENS]
        if not ids:
            ids = [0]
        mask = [1] * len(ids)
        pad = MAX_TOKENS - len(ids)
        if pad > 0:
            ids += [0] * pad
            mask += [0] * pad
        return (
            np.asarray([ids], dtype=np.int64),
            np.asarray([mask], dtype=np.int64),
        )


def _short_hash(path: Path) -> str:
    """用文件内容前 8 MiB 的哈希做标识：整文件哈希 150MB 太慢，够区分权重即可。"""
    import hashlib

    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            digest.update(handle.read(8 * 1024 * 1024))
    except OSError:
        return "unknown"
    return digest.hexdigest()[:12]
