"""别名表：中文查询映射到 CLIP 能懂的英文，且可被配置覆盖。"""

from __future__ import annotations

from neko_image_embed.engine.config import load_settings
from neko_image_embed.engine.lexicon import translate_query


def test_exact_alias() -> None:
    assert translate_query("猫") == "cat"
    assert translate_query("表情包") == "meme sticker"


def test_alias_inside_sentence() -> None:
    translated = translate_query("可爱的猫咪")
    assert "cat" in translated
    assert "可爱" not in translated


def test_english_passes_through() -> None:
    assert translate_query("a red car") == "a red car"


def test_empty_query() -> None:
    assert translate_query("") == ""
    assert translate_query("   ") == ""


def test_extra_overrides_default() -> None:
    assert translate_query("猫", {"猫": "kitty"}) == "kitty"


def test_config_reads_aliases() -> None:
    settings = load_settings({"image_embed": {"query_aliases": {"猫": "cat", "狗": "dog"}}})
    assert settings.query_aliases == {"猫": "cat", "狗": "dog"}


def test_config_ignores_bad_alias_values() -> None:
    settings = load_settings({"image_embed": {"query_aliases": {"猫": 1, "狗": "dog"}}})
    assert settings.query_aliases == {"狗": "dog"}
    assert any("query_aliases" in warning for warning in settings.warnings)


def test_config_rejects_non_table() -> None:
    settings = load_settings({"image_embed": {"query_aliases": ["猫"]}})
    assert settings.query_aliases == {}
    assert any("期望表" in warning for warning in settings.warnings)
