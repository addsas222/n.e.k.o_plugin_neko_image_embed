"""人设解析：配置优先、characters.json 回退、坏文件不阻塞。"""

from __future__ import annotations

import json

from neko_image_embed.engine.persona import load_persona


def _write(tmp_path, payload: object):
    path = tmp_path / "characters.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_config_override_wins(tmp_path) -> None:
    path = _write(tmp_path, {"当前猫娘": "YUI", "猫娘": {"YUI": {"昵称": "YUI"}}})
    persona = load_persona(override="角色：自定义\n很安静", characters_path=path)
    assert persona.source == "config"
    assert "自定义" in persona.text


def test_reads_current_catgirl(tmp_path) -> None:
    path = _write(
        tmp_path,
        {
            "主人": {"昵称": "云朵"},
            "当前猫娘": "YUI",
            "猫娘": {
                "YUI": {
                    "昵称": "YUI",
                    "核心特质": "元气",
                    "行为特点": "爱撒娇",
                    "厌恶": "被忽视",
                }
            },
        },
    )
    persona = load_persona(override="", characters_path=path)
    assert persona.source == "characters"
    assert persona.name == "YUI"
    assert "元气" in persona.text and "云朵" in persona.text


def test_falls_back_to_first_profile_when_current_missing(tmp_path) -> None:
    path = _write(tmp_path, {"猫娘": {"沐莹": {"昵称": "沐莹", "核心特质": "温柔"}}})
    persona = load_persona(override="", characters_path=path)
    assert persona.source == "characters"
    assert persona.name == "沐莹"


def test_missing_file_is_not_fatal(tmp_path) -> None:
    persona = load_persona(override="", characters_path=tmp_path / "nope.json")
    assert persona.source == "none"
    assert persona.warning
    assert "通用陪伴角色" in persona.prompt_block()


def test_broken_json_is_not_fatal(tmp_path) -> None:
    path = tmp_path / "characters.json"
    path.write_text("{ not json", encoding="utf-8")
    persona = load_persona(override="", characters_path=path)
    assert persona.source == "none"
    assert "读取失败" in persona.warning


def test_text_is_clipped(tmp_path) -> None:
    long_text = "角色：" + "很长的设定" * 200
    persona = load_persona(override=long_text, characters_path=None, max_chars=100)
    assert len(persona.text) <= 100
