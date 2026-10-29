"""阶段 4 的落地：SQLite 存取 + 向量检索。

设计取舍：

* **不用 sqlite-vec**。宿主运行时没有 ``vec0`` 扩展（实测确认），所以向量按
  ``float32`` BLOB 存，检索走 numpy 全量余弦。行数上限由 ``max_scan_rows`` 兜住，
  超过就按最近入库截断并如实告警，而不是悄悄变慢。
* **向量空间守卫**：``meta`` 里记 ``embedder_id`` 与维度。换编码器（CLIP → 外部
  embeddings）时维度/空间都变了，必须重建索引；混用会得到看似正常但毫无意义的
  相似度，这类静默错误比崩溃更难查。
* **同步 API**。所有方法都是阻塞的，由插件层用 ``asyncio.to_thread`` 包起来。
  这样单测不需要事件循环，也不会在命令循环里卡住别的入口。
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
MAX_EVENTS = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL,
    pixel_sha TEXT,
    lanlan TEXT NOT NULL DEFAULT '',
    rel_path TEXT NOT NULL,
    thumb_path TEXT,
    width INTEGER,
    height INTEGER,
    bytes INTEGER,
    mime TEXT,
    caption TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    sticker INTEGER DEFAULT 0,
    verdict TEXT DEFAULT '',
    engine TEXT DEFAULT '',
    modality TEXT DEFAULT '',
    value REAL DEFAULT 0,
    risk_max REAL DEFAULT 0,
    reason TEXT DEFAULT '',
    dhash INTEGER DEFAULT 0,
    vector BLOB,
    dim INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    used_at REAL,
    use_count INTEGER DEFAULT 0,
    deleted INTEGER DEFAULT 0,
    deleted_at REAL,
    UNIQUE(sha256, lanlan)
);
CREATE INDEX IF NOT EXISTS idx_images_live ON images(deleted, lanlan, created_at DESC);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL DEFAULT '',
    level TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL DEFAULT '',
    image_id INTEGER DEFAULT 0,
    engine TEXT DEFAULT '',
    ms REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
"""


class StoreError(RuntimeError):
    pass


class EmbedderMismatch(StoreError):
    """库里的向量空间与当前编码器不一致。"""


@dataclass(frozen=True)
class ImageRow:
    id: int
    sha256: str
    pixel_sha: str
    lanlan: str
    rel_path: str
    thumb_path: str
    width: int
    height: int
    bytes: int
    mime: str
    caption: str
    tags: list[str]
    sticker: bool
    verdict: str
    engine: str
    modality: str
    value: float
    risk_max: float
    reason: str
    dhash: int
    created_at: float
    used_at: float | None
    use_count: int


def encode_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


_U64 = 1 << 64
_I63 = 1 << 63


def to_signed64(value: int) -> int:
    """SQLite 的 INTEGER 是有符号 64 位；dHash 是无符号 64 位，必须折一下。

    不折的话，``1 << 63`` 以上的哈希会直接 OverflowError，且只在「碰巧」的图上炸。
    """
    value &= _U64 - 1
    return value - _U64 if value >= _I63 else value


def from_signed64(value: int) -> int:
    return int(value) & (_U64 - 1)


_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF), (0xAC00, 0xD7AF))


def _has_cjk(text: str) -> bool:
    return any(lo <= ord(char) <= hi for char in text for lo, hi in _CJK_RANGES)


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().replace("\n", " ").split())


def lexical_score(query: str, caption: str, tags: list[str]) -> float:
    """中文/英文词法匹配分（0–1）。

    存在的理由：CLIP ViT-B/32 的文本塔以英文为主，实测「一只狗」与狗的照片余弦只有
    0.20，和无关图拉不开差距。而 caption/tags 往往就是中文——用字面匹配兜住这一段，
    比换多语种编码器便宜得多。
    """
    needle = _normalize(query)
    if not needle:
        return 0.0
    haystack = _normalize(caption) + " " + _normalize(" ".join(tags))
    if not haystack.strip():
        return 0.0

    if _has_cjk(needle):
        compact = needle.replace(" ", "")
        grams = {compact[i : i + 2] for i in range(len(compact) - 1)} or {compact}
        hits = sum(1 for gram in grams if gram in haystack)
        return hits / len(grams)

    tokens = [token for token in needle.split() if len(token) > 1]
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in haystack) / len(tokens)


def _row_tags(row: sqlite3.Row) -> list[str]:
    try:
        tags = json.loads(row["tags"] or "[]")
    except (ValueError, TypeError):
        return []
    return [str(tag) for tag in tags] if isinstance(tags, list) else []


def _row_to_image(row: sqlite3.Row) -> ImageRow:
    try:
        tags = json.loads(row["tags"] or "[]")
    except ValueError:
        tags = []
    if not isinstance(tags, list):
        tags = []
    return ImageRow(
        id=int(row["id"]),
        sha256=str(row["sha256"]),
        pixel_sha=str(row["pixel_sha"] or ""),
        lanlan=str(row["lanlan"] or ""),
        rel_path=str(row["rel_path"]),
        thumb_path=str(row["thumb_path"] or ""),
        width=int(row["width"] or 0),
        height=int(row["height"] or 0),
        bytes=int(row["bytes"] or 0),
        mime=str(row["mime"] or "image/jpeg"),
        caption=str(row["caption"] or ""),
        tags=[str(tag) for tag in tags],
        sticker=bool(row["sticker"]),
        verdict=str(row["verdict"] or ""),
        engine=str(row["engine"] or ""),
        modality=str(row["modality"] or ""),
        value=float(row["value"] or 0.0),
        risk_max=float(row["risk_max"] or 0.0),
        reason=str(row["reason"] or ""),
        dhash=int(from_signed64(int(row["dhash"] or 0))),
        created_at=float(row["created_at"] or 0.0),
        used_at=None if row["used_at"] is None else float(row["used_at"]),
        use_count=int(row["use_count"] or 0),
    )


class ImageStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.embedder_id = ""
        self.dim = 0

    # ------------------------------------------------------------------ 生命周期

    def open(self, *, embedder_id: str, dim: int, persona_name: str = "") -> None:
        """建库并与当前编码器对齐。空间不一致时抛 EmbedderMismatch。"""
        if not embedder_id or dim <= 0:
            raise ValueError("embedder_id 与 dim 必须有效")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn
        self._lock = threading.RLock()

        meta = self._read_meta()
        stored_id = meta.get("embedder_id", "")
        stored_dim = int(meta.get("dim", "0") or 0)
        if stored_id and (stored_id != embedder_id or stored_dim != dim):
            self.close()
            raise EmbedderMismatch(
                "索引是用另一个编码器建的："
                f"{stored_id}({stored_dim}维) != {embedder_id}({dim}维)。"
                "换编码器必须重建索引（img_gc(hard=true) 后重新收录）。"
            )
        if not stored_id:
            with self._lock:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("embedder_id", embedder_id),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("dim", str(dim)),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
                conn.commit()
        if persona_name:
            with self._lock:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    ("persona_name", persona_name),
                )
                conn.commit()
        self.embedder_id = stored_id or embedder_id
        self.dim = stored_dim or dim

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.commit()
                self._conn.close()
                self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StoreError("数据库尚未打开")
        return self._conn

    def _read_meta(self) -> dict[str, str]:
        try:
            rows = self.conn.execute("SELECT key, value FROM meta").fetchall()
        except sqlite3.Error:
            return {}
        return {str(row["key"]): str(row["value"]) for row in rows}

    # ------------------------------------------------------------------ 写入

    def add(
        self,
        *,
        sha256: str,
        pixel_sha: str,
        lanlan: str,
        rel_path: str,
        thumb_path: str,
        width: int,
        height: int,
        size_bytes: int,
        mime: str,
        caption: str,
        tags: list[str],
        sticker: bool,
        verdict: str,
        engine: str,
        modality: str,
        value: float,
        risk_max: float,
        reason: str,
        dhash: int,
        vector: np.ndarray,
    ) -> tuple[int, bool]:
        """返回 (id, created)。同 (sha256, lanlan) 已存在时返回旧 id 且 created=False。"""
        if vector.size != self.dim:
            raise StoreError(f"向量维度 {vector.size} 与索引 {self.dim} 不一致")
        payload = encode_vector(vector)
        now = time.time()
        with self._lock:
            try:
                cursor = self.conn.execute(
                    """
                    INSERT INTO images (
                        sha256, pixel_sha, lanlan, rel_path, thumb_path, width, height, bytes,
                        mime, caption, tags, sticker, verdict, engine, modality, value,
                        risk_max, reason, dhash, vector, dim, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        sha256, pixel_sha, lanlan, rel_path, thumb_path, width, height,
                        size_bytes, mime, caption, json.dumps(tags, ensure_ascii=False),
                        int(sticker), verdict, engine, modality, float(value), float(risk_max),
                        reason, to_signed64(int(dhash)), payload, int(self.dim), now,
                    ),
                )
                self.conn.commit()
                return int(cursor.lastrowid or 0), True
            except sqlite3.IntegrityError:
                row = self.conn.execute(
                    "SELECT id FROM images WHERE sha256=? AND lanlan=?", (sha256, lanlan)
                ).fetchone()
                return (int(row["id"]) if row else 0), False

    def touch(self, image_id: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE images SET used_at=?, use_count=use_count+1 WHERE id=?",
                (time.time(), image_id),
            )
            self.conn.commit()

    def soft_delete(self, image_id: int) -> bool:
        with self._lock:
            cursor = self.conn.execute(
                "UPDATE images SET deleted=1, deleted_at=? WHERE id=? AND deleted=0",
                (time.time(), image_id),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ 查询

    def get(self, image_id: int) -> ImageRow | None:
        row = self.conn.execute(
            "SELECT * FROM images WHERE id=? AND deleted=0", (image_id,)
        ).fetchone()
        return _row_to_image(row) if row else None

    def find_by_sha(self, sha256: str, lanlan: str) -> ImageRow | None:
        row = self.conn.execute(
            "SELECT * FROM images WHERE sha256=? AND lanlan=? AND deleted=0",
            (sha256, lanlan),
        ).fetchone()
        return _row_to_image(row) if row else None

    def find_near_duplicate(
        self, dhash_value: int, lanlan: str, *, max_hamming: int, scan: int = 5000
    ) -> ImageRow | None:
        """dHash 近重复：只扫最近 ``scan`` 条，避免全表 O(n)。"""
        rows = self.conn.execute(
            "SELECT * FROM images WHERE deleted=0 AND lanlan=? ORDER BY created_at DESC LIMIT ?",
            (lanlan, int(scan)),
        ).fetchall()
        best: tuple[int, sqlite3.Row] | None = None
        target = from_signed64(dhash_value)
        for row in rows:
            distance = (target ^ from_signed64(int(row["dhash"] or 0))).bit_count()
            if distance <= max_hamming and (best is None or distance < best[0]):
                best = (distance, row)
        return _row_to_image(best[1]) if best else None

    def search(
        self,
        vector: np.ndarray,
        *,
        lanlan: str,
        top_k: int,
        max_scan_rows: int,
        query_text: str = "",
    ) -> tuple[list[tuple[ImageRow, float]], bool]:
        """返回 ([(row, 分数)], truncated)。

        分数默认是余弦；给了 ``query_text`` 时，对 caption/tags 命中的行做词法加权
        （``0.6*cos + 0.4*lex``），用来兜住 CLIP 文本塔对中文的弱势。
        """
        if vector.size != self.dim:
            raise StoreError(f"查询向量维度 {vector.size} 与索引 {self.dim} 不一致")
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm <= 1e-8:
            raise StoreError("查询向量为零向量（空文本或解码失败）")
        query = query / norm

        clause = "deleted=0"
        params: list[object] = []
        if lanlan:
            clause += " AND lanlan=?"
            params.append(lanlan)
        rows = self.conn.execute(
            f"SELECT * FROM images WHERE {clause} ORDER BY created_at DESC LIMIT ?",
            (*params, int(max_scan_rows) + 1),
        ).fetchall()
        truncated = len(rows) > max_scan_rows
        rows = rows[:max_scan_rows]
        if not rows:
            return [], truncated

        ids: list[int] = []
        vectors: list[np.ndarray] = []
        kept: list[sqlite3.Row] = []
        for row in rows:
            blob = row["vector"]
            if not blob:
                continue
            candidate = decode_vector(blob)
            if candidate.size != query.size:
                continue
            ids.append(int(row["id"]))
            vectors.append(candidate)
            kept.append(row)

        if not vectors:
            return [], truncated

        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        scores = (matrix / norms) @ query

        if query_text.strip():
            lexical = np.fromiter(
                (
                    lexical_score(query_text, str(row["caption"] or ""), _row_tags(row))
                    for row in kept
                ),
                dtype=np.float32,
                count=len(kept),
            )
            boosted = 0.6 * scores + 0.4 * lexical
            scores = np.where(lexical > 0, boosted, scores)

        order = np.argsort(-scores)[: max(1, top_k)]

        by_id = {int(row["id"]): row for row in kept}
        results: list[tuple[ImageRow, float]] = []
        for index in order:
            row = by_id.get(ids[int(index)])
            if row is not None:
                results.append((_row_to_image(row), float(scores[int(index)])))
        return results, truncated

    def recent(self, *, lanlan: str, limit: int) -> list[ImageRow]:
        clause = "deleted=0"
        params: list[object] = []
        if lanlan:
            clause += " AND lanlan=?"
            params.append(lanlan)
        rows = self.conn.execute(
            f"SELECT * FROM images WHERE {clause} ORDER BY created_at DESC LIMIT ?",
            (*params, int(limit)),
        ).fetchall()
        return [_row_to_image(row) for row in rows]

    # ------------------------------------------------------------------ 回收

    def take_deleted(self, *, before: float, limit: int) -> list[ImageRow]:
        rows = self.conn.execute(
            "SELECT * FROM images WHERE deleted=1 AND deleted_at IS NOT NULL AND deleted_at<=? "
            "ORDER BY deleted_at ASC LIMIT ?",
            (before, int(limit)),
        ).fetchall()
        return [_row_to_image(row) for row in rows]

    def take_oldest(self, *, keep: int, limit: int) -> list[ImageRow]:
        rows = self.conn.execute(
            "SELECT * FROM images WHERE deleted=0 AND id NOT IN "
            "(SELECT id FROM images WHERE deleted=0 ORDER BY created_at DESC LIMIT ?) "
            "ORDER BY created_at ASC LIMIT ?",
            (int(keep), int(limit)),
        ).fetchall()
        return [_row_to_image(row) for row in rows]

    def hard_delete(self, image_id: int) -> bool:
        with self._lock:
            cursor = self.conn.execute("DELETE FROM images WHERE id=?", (image_id,))
            self.conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ 统计

    # ------------------------------------------------------------------ 事件日志

    def log_event(
        self,
        kind: str,
        message: str,
        *,
        level: str = "info",
        image_id: int = 0,
        engine: str = "",
        ms: float = 0.0,
    ) -> None:
        """写一条活动记录并做环形修剪（保留最近 MAX_EVENTS 条）。"""
        with self._lock:
            if self._conn is None:
                raise StoreError("索引未打开")
            cursor = self._conn.execute(
                "INSERT INTO events(ts, kind, level, message, image_id, engine, ms) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    time.time(),
                    str(kind)[:32],
                    str(level)[:16],
                    str(message)[:500],
                    int(image_id or 0),
                    str(engine)[:32],
                    float(ms or 0.0),
                ),
            )
            last_id = int(cursor.lastrowid or 0)
            # 每 50 条修剪一次，避免每次写都跑子查询
            if last_id and last_id % 50 == 0:
                self._conn.execute("DELETE FROM events WHERE id <= ?", (last_id - MAX_EVENTS,))
            self._conn.commit()

    def recent_events(self, *, limit: int = 40, level: str = "") -> list[dict[str, object]]:
        clause = ""
        params: list[object] = []
        if level:
            clause = "WHERE level=?"
            params.append(level)
        params.append(max(1, int(limit)))
        rows = self._conn.execute(
            f"SELECT id, ts, kind, level, message, image_id, engine, ms "
            f"FROM events {clause} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def export_row(self, image_id: int) -> dict[str, object]:
        """导出一条完整记录（含向量），用于跨档迁移。"""
        row = self._conn.execute(
            "SELECT * FROM images WHERE id=? AND deleted=0", (int(image_id),)
        ).fetchone()
        if row is None:
            raise StoreError(f"#{image_id} 不存在或已删除")
        data = dict(row)
        blob = data.pop("vector", None)
        data["vector"] = decode_vector(blob) if blob else np.zeros(self.dim, dtype=np.float32)
        return data

    def import_row(self, data: dict[str, object]) -> tuple[int, bool]:
        """把 export_row 的产物写进当前档（同一张图可指向新的 lanlan）。"""
        return self.add(
            sha256=str(data["sha256"]),
            pixel_sha=str(data.get("pixel_sha") or ""),
            lanlan=str(data.get("lanlan") or ""),
            rel_path=str(data["rel_path"]),
            thumb_path=str(data.get("thumb_path") or ""),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            size_bytes=int(data.get("bytes") or 0),
            mime=str(data.get("mime") or "image/jpeg"),
            caption=str(data.get("caption") or ""),
            tags=list(data.get("tags") or []),
            sticker=bool(data.get("sticker")),
            verdict=str(data.get("verdict") or "accepted"),
            engine=str(data.get("engine") or ""),
            modality=str(data.get("modality") or ""),
            value=float(data.get("value") or 0.0),
            risk_max=float(data.get("risk_max") or 0.0),
            reason=str(data.get("reason") or ""),
            dhash=int(data.get("dhash") or 0),
            vector=np.asarray(data["vector"], dtype=np.float32),
        )

    def set_lanlan(self, image_id: int, lanlan: str) -> bool:
        """改归属。用于把「未归属」的旧记录收回到某个角色名下。"""
        target = (lanlan or "").strip()
        if not target:
            raise StoreError("归属角色不能为空")
        with self._lock:
            if self._conn is None:
                raise StoreError("索引未打开")
            try:
                cursor = self._conn.execute(
                    "UPDATE images SET lanlan=? WHERE id=? AND deleted=0", (target, int(image_id))
                )
            except sqlite3.IntegrityError as exc:
                raise StoreError(f"该角色下已存在同一张图：{exc}") from exc
            self._conn.commit()
            return cursor.rowcount > 0

    def stats(self) -> dict[str, object]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN deleted=1 THEN 1 ELSE 0 END) AS deleted, "
            "SUM(bytes) AS bytes FROM images"
        ).fetchone()
        total = int(row["total"] or 0)
        deleted = int(row["deleted"] or 0)
        return {
            "total": total,
            "live": total - deleted,
            "deleted": deleted,
            "bytes": int(row["bytes"] or 0),
            "embedder_id": self.embedder_id,
            "dim": self.dim,
        }
