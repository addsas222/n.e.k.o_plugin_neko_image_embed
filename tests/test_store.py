"""存储测试：去重、检索、空间守卫、软删。"""

from __future__ import annotations

import numpy as np
import pytest
from neko_image_embed.engine.store import (
    EmbedderMismatch,
    ImageStore,
    StoreError,
    lexical_score,
)


def _vectors() -> list[np.ndarray]:
    base = np.zeros(8, dtype=np.float32)
    out = []
    for index in range(3):
        vector = base.copy()
        vector[index] = 1.0
        out.append(vector)
    return out


def _add(store: ImageStore, vector: np.ndarray, sha: str, *, lanlan: str = "YUI") -> tuple[int, bool]:
    return store.add(
        sha256=sha,
        pixel_sha=sha,
        lanlan=lanlan,
        rel_path=f"media/{sha}.jpg",
        thumb_path="",
        width=100,
        height=80,
        size_bytes=1234,
        mime="image/jpeg",
        caption=f"caption {sha}",
        tags=["tag"],
        sticker=False,
        verdict="accepted",
        engine="rules",
        modality="heuristic",
        value=0.5,
        risk_max=0.0,
        reason="test",
        dhash=0,
        vector=vector,
    )


def test_add_and_search(tmp_path) -> None:
    store = ImageStore(tmp_path / "index.db")
    store.open(embedder_id="fake-1", dim=8)
    try:
        first, created = _add(store, _vectors()[0], "a" * 64)
        assert created is True
        _add(store, _vectors()[1], "b" * 64)

        results, truncated = store.search(
            _vectors()[0], lanlan="YUI", top_k=2, max_scan_rows=100
        )
        assert truncated is False
        assert results[0][0].id == first
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    finally:
        store.close()


def test_duplicate_returns_existing_id(tmp_path) -> None:
    store = ImageStore(tmp_path / "index.db")
    store.open(embedder_id="fake-1", dim=8)
    try:
        first, created = _add(store, _vectors()[0], "a" * 64)
        second, created_again = _add(store, _vectors()[1], "a" * 64)
        assert created is True and created_again is False
        assert first == second
        assert store.stats()["total"] == 1
    finally:
        store.close()


def test_embedder_mismatch_is_refused(tmp_path) -> None:
    db = tmp_path / "index.db"
    store = ImageStore(db)
    store.open(embedder_id="fake-1", dim=8)
    _add(store, _vectors()[0], "a" * 64)
    store.close()

    other = ImageStore(db)
    with pytest.raises(EmbedderMismatch):
        other.open(embedder_id="fake-2", dim=8)
    other.close()


def test_dimension_mismatch_on_add(tmp_path) -> None:
    store = ImageStore(tmp_path / "index.db")
    store.open(embedder_id="fake-1", dim=8)
    try:
        with pytest.raises(StoreError):
            _add(store, np.zeros(4, dtype=np.float32), "c" * 64)
    finally:
        store.close()


def test_zero_vector_query_is_refused(tmp_path) -> None:
    store = ImageStore(tmp_path / "index.db")
    store.open(embedder_id="fake-1", dim=8)
    try:
        _add(store, _vectors()[0], "a" * 64)
        with pytest.raises(StoreError):
            store.search(np.zeros(8, dtype=np.float32), lanlan="", top_k=3, max_scan_rows=10)
    finally:
        store.close()


def test_soft_delete_hides_from_search(tmp_path) -> None:
    store = ImageStore(tmp_path / "index.db")
    store.open(embedder_id="fake-1", dim=8)
    try:
        image_id, _ = _add(store, _vectors()[0], "a" * 64)
        assert store.soft_delete(image_id) is True
        assert store.get(image_id) is None
        results, _ = store.search(_vectors()[0], lanlan="", top_k=3, max_scan_rows=10)
        assert results == []
        stats = store.stats()
        assert stats["live"] == 0 and stats["deleted"] == 1
    finally:
        store.close()


def test_truncation_flag(tmp_path) -> None:
    store = ImageStore(tmp_path / "index.db")
    store.open(embedder_id="fake-1", dim=8)
    try:
        for index in range(3):
            _add(store, _vectors()[index], f"{index}" * 64)
        _results, truncated = store.search(
            _vectors()[0], lanlan="", top_k=3, max_scan_rows=2
        )
        assert truncated is True
    finally:
        store.close()


def test_persona_archives_are_physically_separate(tmp_path) -> None:
    """每个角色一套独立存档：不同 index.db、不同目录、互不可见。"""
    from neko_image_embed.engine.personas import PersonaArchives

    archives = PersonaArchives(tmp_path / "images", embedder_id="fake-1", dim=8)
    try:
        yui = archives.store_for("YUI")
        mio = archives.store_for("Mio")
        assert yui.db_path != mio.db_path
        assert yui.db_path.parent.name == "YUI"
        assert mio.db_path.parent.name == "Mio"

        image_id, created = _add(yui, _vectors()[0], "a" * 64, lanlan="YUI")
        assert created is True
        assert mio.stats()["total"] == 0, "另一档不该出现这张图"
        assert mio.find_by_sha("a" * 64, "Mio") is None

        # 同一张图可以在另一档独立存在（各有各的 value/caption）；id 只在档内唯一
        other_id, other_created = _add(mio, _vectors()[1], "a" * 64, lanlan="Mio")
        assert other_created is True
        assert other_id == image_id, "两档各自从 1 开始编号，id 允许相同"
        assert yui.stats()["total"] == 1 and mio.stats()["total"] == 1

        # 迁移：连文件一起搬到目标档，源档删除
        media = yui.db_path.parent / "media"
        media.mkdir(parents=True, exist_ok=True)
        (media / f"{'a' * 64}.jpg").write_bytes(b"jpeg")
        moved_id = archives.move(from_persona="YUI", image_id=image_id, to_persona="Mio2")
        assert yui.stats()["live"] == 0
        moved = archives.store_for("Mio2").get(moved_id)
        assert moved is not None and moved.lanlan == "Mio2"
        assert (archives.root_for("Mio2") / "media" / f"{'a' * 64}.jpg").is_file()
        assert not (media / f"{'a' * 64}.jpg").exists(), "源档文件应被移走"
    finally:
        archives.close()


def test_persona_slug_keeps_names_readable(tmp_path) -> None:
    from neko_image_embed.engine.personas import persona_slug

    assert persona_slug("YUI") == "YUI"
    assert persona_slug("沐莹") == "沐莹"
    assert "/" not in persona_slug("a/b:c*?")
    assert persona_slug("   ") == "default"


def test_set_lanlan_moves_row(tmp_path) -> None:
    store = ImageStore(tmp_path / "index.db")
    store.open(embedder_id="fake-1", dim=8)
    try:
        image_id, _ = _add(store, _vectors()[0], "a" * 64, lanlan="YUI")
        assert store.get(image_id).lanlan == "YUI"
        assert store.set_lanlan(image_id, "Mio") is True
        assert store.get(image_id).lanlan == "Mio"
        results, _ = store.search(_vectors()[0], lanlan="YUI", top_k=3, max_scan_rows=10)
        assert results == [], "改归属后不该再出现在原角色视野里"
        assert store.set_lanlan(9999, "Mio") is False
        with pytest.raises(StoreError):
            store.set_lanlan(image_id, "  ")
    finally:
        store.close()


def test_events_are_bounded_and_filterable(tmp_path, monkeypatch) -> None:
    from neko_image_embed.engine import store as store_mod

    store = ImageStore(tmp_path / "index.db")
    store.open(embedder_id="fake-1", dim=8)
    try:
        for index in range(5):
            store.log_event(
                "ingest",
                f"m{index}",
                level="info" if index % 2 == 0 else "error",
                image_id=index,
                engine="laya",
                ms=12.5,
            )
        rows = store.recent_events(limit=3)
        assert [row["message"] for row in rows] == ["m4", "m3", "m2"]
        assert rows[0]["engine"] == "laya" and rows[0]["ms"] == 12.5

        errors = store.recent_events(limit=10, level="error")
        assert {row["level"] for row in errors} == {"error"}

        monkeypatch.setattr(store_mod, "MAX_EVENTS", 10)
        for index in range(60):
            store.log_event("x", f"n{index}")
        total = store.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert total <= store_mod.MAX_EVENTS + 50, "环形修剪没生效"
    finally:
        store.close()


def test_lexical_score_handles_cjk() -> None:
    assert lexical_score("猫", "一只橘猫趴在键盘上", []) > 0.5
    assert lexical_score("猫", "蓝色汽车", ["交通"]) == 0.0
    assert lexical_score("red car", "a red car on the road", []) == 1.0
    assert lexical_score("", "任意 caption", []) == 0.0


def test_lexical_boost_rescues_caption_hits(tmp_path) -> None:
    """CLIP 文本塔对中文很弱；caption 命中要能把低余弦的行拉上来。"""
    store = ImageStore(tmp_path / "index.db")
    store.open(embedder_id="fake-1", dim=8)
    try:
        base = _vectors()[0]
        other = _vectors()[1]
        car_vec = base                      # 与查询完全同向：cos = 1
        cat_vec = other                     # 正交：cos ≈ 0
        dog_vec = 0.2 * base + 0.98 * other  # cos ≈ 0.2

        cat_id, _ = _add(store, cat_vec, "a" * 64)
        car_id, _ = _add(store, car_vec, "b" * 64)
        dog_id, _ = _add(store, dog_vec, "e" * 64)

        rows = {
            cat_id: ("一只橘猫", '["猫"]'),
            car_id: ("蓝色汽车", '["车"]'),
            dog_id: ("一只柴犬", '["狗"]'),
        }
        for image_id, (caption, tags) in rows.items():
            store.conn.execute(
                "UPDATE images SET caption=?, tags=? WHERE id=?", (caption, tags, image_id)
            )
        store.conn.commit()

        results, _ = store.search(
            base, lanlan="YUI", top_k=3, max_scan_rows=100, query_text="猫"
        )
        order = [row.id for row, _score in results]
        assert order[0] == car_id              # 余弦满分仍排第一
        assert order.index(cat_id) < order.index(dog_id)  # caption 命中被拉起来

        plain, _ = store.search(base, lanlan="YUI", top_k=3, max_scan_rows=100)
        assert [row.id for row, _score in plain][0] == car_id
    finally:
        store.close()
