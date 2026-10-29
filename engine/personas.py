"""按角色分档：每个人设一套完全独立的存档。

为什么不是「一张表加一列 lanlan」：那样只是查询时过滤，档案本身还是共用的——
换角色看到的是同一个库、同一份文件目录，任何一处漏了过滤就串档。
这里改成物理隔离：

```
data/images/
├── YUI/
│   ├── index.db          # 只属于 YUI 的元数据 + 向量
│   ├── media/<sha>.jpg
│   └── thumbs/<sha>.jpg
└── 沐莹/
    └── ...
```

每个档有自己的 ``embedder_id``/``dim``/``persona_name`` 元数据，互不引用；
同一张图在两个角色下各存一份、各有各的 value 与 caption（需求度天然按人设）。
"""

from __future__ import annotations

import contextlib
import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .store import ImageStore, StoreError

_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def persona_slug(name: str) -> str:
    """目录名：保留可读性，去掉文件系统非法字符。"""
    cleaned = _ILLEGAL.sub("_", (name or "").strip()).strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned or "default")[:40]


@dataclass(frozen=True)
class ArchiveInfo:
    persona: str
    slug: str
    root: str
    live: int
    total: int
    bytes: int


class PersonaArchives:
    """角色 → 档案目录 的映射，负责惰性开档、迁移和统一关闭。"""

    def __init__(self, root: Path, *, embedder_id: str, dim: int) -> None:
        self._root = Path(root)
        self._embedder_id = embedder_id
        self._dim = dim
        self._stores: dict[str, ImageStore] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 路径

    def root_for(self, persona: str) -> Path:
        return self._root / persona_slug(persona)

    def media_root(self, persona: str) -> Path:
        return self.root_for(persona)

    @property
    def legacy_root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------ 开档

    def store_for(self, persona: str) -> ImageStore:
        persona = (persona or "").strip()
        if not persona:
            raise StoreError("缺少角色名，无法定位存档")
        with self._lock:
            store = self._stores.get(persona)
            if store is not None:
                return store
            root = self.root_for(persona)
            store = ImageStore(root / "index.db")
            store.open(
                embedder_id=self._embedder_id,
                dim=self._dim,
                persona_name=persona,
            )
            self._stores[persona] = store
            return store

    def close(self) -> None:
        with self._lock:
            for store in self._stores.values():
                with contextlib.suppress(Exception):
                    store.close()
            self._stores.clear()

    # ------------------------------------------------------------------ 枚举

    def list_archives(self) -> list[ArchiveInfo]:
        if not self._root.is_dir():
            return []
        out: list[ArchiveInfo] = []
        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            db = child / "index.db"
            if not db.is_file():
                continue
            try:
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
                    con.row_factory = sqlite3.Row
                    meta = {
                        row["key"]: row["value"]
                        for row in con.execute("SELECT key, value FROM meta")
                    }
                    row = con.execute(
                        "SELECT COUNT(*) AS total, "
                        "SUM(CASE WHEN deleted=1 THEN 1 ELSE 0 END) AS deleted, "
                        "SUM(bytes) AS bytes FROM images"
                    ).fetchone()
            except sqlite3.Error:
                continue
            total = int(row["total"] or 0)
            deleted = int(row["deleted"] or 0)
            out.append(
                ArchiveInfo(
                    persona=meta.get("persona_name") or child.name,
                    slug=child.name,
                    root=str(child),
                    live=total - deleted,
                    total=total,
                    bytes=int(row["bytes"] or 0),
                )
            )
        return out

    # ------------------------------------------------------------------ 跨档迁移

    def move(self, *, from_persona: str, image_id: int, to_persona: str) -> int:
        """把一条记录连同文件搬到另一个角色的存档里。源档删除，目标档新建。"""
        target = (to_persona or "").strip()
        if not target:
            raise StoreError("目标角色不能为空")
        if persona_slug(target) == persona_slug(from_persona):
            raise StoreError("源与目标角色相同")
        src = self.store_for(from_persona)
        dst = self.store_for(target)
        data = src.export_row(image_id)
        data["lanlan"] = target

        src_root = self.root_for(from_persona)
        dst_root = self.root_for(target)
        copied: list[Path] = []
        for rel in (str(data["rel_path"]), str(data.get("thumb_path") or "")):
            if not rel:
                continue
            source = src_root / rel
            if not source.is_file():
                raise StoreError(f"源文件缺失：{rel}")
            dest = dst_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(dest)

        new_id, created = dst.import_row(data)
        if not created:
            for path in copied:
                path.unlink(missing_ok=True)
            raise StoreError("目标档里已存在同一张图")
        src.hard_delete(image_id)
        for rel in (str(data["rel_path"]), str(data.get("thumb_path") or "")):
            if rel:
                (src_root / rel).unlink(missing_ok=True)
        return new_id

    # ------------------------------------------------------------------ 迁移

    def migrate_legacy(self, *, default_persona: str) -> dict[str, object]:
        """把旧版「单库 + lanlan 列」迁移成按角色分档。

        旧布局：``data/images/{index.db,media/,thumbs/}``。
        迁移后旧库改名 ``index.db.migrated-<ts>``，文件按行搬到各角色目录；
        搬成功的文件从旧目录删掉，失败的原样留下并计入 failures。
        """
        legacy_db = self._root / "index.db"
        if not legacy_db.is_file():
            return {"migrated": 0, "failures": [], "skipped": "no legacy db"}

        moved = 0
        failures: list[str] = []
        try:
            con = sqlite3.connect(f"file:{legacy_db}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id, sha256, pixel_sha, lanlan, rel_path, thumb_path, width, height, "
                "bytes, mime, caption, tags, sticker, verdict, engine, modality, value, "
                "risk_max, reason, dhash, vector, dim, created_at FROM images WHERE deleted=0"
            ).fetchall()
            con.close()
        except sqlite3.Error as exc:
            return {"migrated": 0, "failures": [f"读取旧库失败：{exc}"], "skipped": ""}

        import json

        import numpy as np

        for row in rows:
            persona = (row["lanlan"] or "").strip() or default_persona
            if not persona:
                failures.append(f"#{row['id']}: 没有 lanlan，且无默认角色")
                continue
            root = self.root_for(persona)
            media = root / (row["rel_path"] or "")
            thumb = root / (row["thumb_path"] or "") if row["thumb_path"] else None
            src_media = self._root / (row["rel_path"] or "")
            src_thumb = self._root / (row["thumb_path"] or "") if row["thumb_path"] else None

            try:
                store = self.store_for(persona)
                blob = row["vector"]
                vector = (
                    np.frombuffer(blob, dtype=np.float32).copy()
                    if blob
                    else np.zeros(self._dim, dtype=np.float32)
                )
                image_id, created = store.add(
                    sha256=row["sha256"],
                    pixel_sha=row["pixel_sha"] or "",
                    lanlan=persona,
                    rel_path=row["rel_path"],
                    thumb_path=row["thumb_path"] or "",
                    width=int(row["width"] or 0),
                    height=int(row["height"] or 0),
                    size_bytes=int(row["bytes"] or 0),
                    mime=row["mime"] or "image/jpeg",
                    caption=row["caption"] or "",
                    tags=json.loads(row["tags"] or "[]"),
                    sticker=bool(row["sticker"]),
                    verdict=row["verdict"] or "accepted",
                    engine=row["engine"] or "",
                    modality=row["modality"] or "",
                    value=float(row["value"] or 0),
                    risk_max=float(row["risk_max"] or 0),
                    reason=row["reason"] or "",
                    dhash=int(row["dhash"] or 0),
                    vector=vector,
                )
                if not created:
                    failures.append(f"#{row['id']}: 目标档里已存在同一张图（{persona}）")
                    continue
                media.parent.mkdir(parents=True, exist_ok=True)
                if src_media.is_file():
                    shutil.copy2(src_media, media)
                    src_media.unlink(missing_ok=True)
                if thumb is not None and src_thumb is not None and src_thumb.is_file():
                    thumb.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_thumb, thumb)
                    src_thumb.unlink(missing_ok=True)
                moved += 1
            except (StoreError, OSError, ValueError) as exc:
                failures.append(f"#{row['id']}: {type(exc).__name__}: {exc}")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        with contextlib.suppress(sqlite3.Error), sqlite3.connect(str(legacy_db)) as con:
            # 先把 WAL 并回主库，避免改名后 sidecar 成为孤儿
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        try:
            legacy_db.rename(self._root / f"index.db.migrated-{stamp}")
        except OSError as exc:
            failures.append(f"旧库改名失败：{exc}")

        for suffix in ("-wal", "-shm"):
            (self._root / f"index.db{suffix}").unlink(missing_ok=True)

        for leftover in ("media", "thumbs"):
            path = self._root / leftover
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

        return {"migrated": moved, "failures": failures[:20], "skipped": ""}
