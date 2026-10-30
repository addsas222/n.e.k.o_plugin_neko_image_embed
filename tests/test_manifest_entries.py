"""清单与装饰器对账：入口以代码里的 @plugin_entry 为唯一事实来源。

宿主重扫插件时只认 plugin.toml / plugin.meta.json 里的静态声明，少写一条，
``/plugins/refresh`` 之后对应 action 就 404。所以这条测试是硬门槛。
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _declared_entries() -> dict[str, str]:
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if not (isinstance(deco, ast.Call) and getattr(deco.func, "id", None) == "plugin_entry"):
                continue
            kwargs = {kw.arg: kw.value for kw in deco.keywords}
            entry_id = kwargs["id"].value if "id" in kwargs else node.name
            # name 现在可能是 tr("entries.x.name", default="...") 的调用
            name_node = kwargs.get("name")
            if isinstance(name_node, ast.Constant):
                name = str(name_node.value)
            elif isinstance(name_node, ast.Call):
                defaults = [
                    str(kw.value.value)
                    for kw in name_node.keywords
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant)
                ]
                name = defaults[0] if defaults else ""
            else:
                name = ""
            found[str(entry_id)] = name
    return found


def _manifest() -> dict:
    return tomllib.loads((ROOT / "plugin.toml").read_text(encoding="utf-8"))


def test_manifest_declares_every_decorated_entry() -> None:
    # plugin.toml 不再重复声明 entries（schema 不认这个字段，宿主从装饰器收集）
    declared = _declared_entries()
    assert {"img_ingest", "img_search", "img_send", "img_forget", "img_gc", "img_status"} <= set(declared)


def test_plugin_id_matches_directory() -> None:
    manifest = _manifest()["plugin"]
    assert manifest["id"] == ROOT.name


def test_entry_module_matches_plugin_id() -> None:
    manifest = _manifest()["plugin"]
    module, _, cls = manifest["entry"].partition(":")
    assert module == f"plugin.plugins.{ROOT.name}"
    assert cls == "NekoImageEmbedPlugin"


def test_sdk_range_is_sane() -> None:
    sdk = _manifest()["plugin"]["sdk"]
    assert sdk["supported"].startswith(">=0.1.0")
    assert "0.1.0" in sdk["recommended"]


def test_config_example_has_no_plugin_section() -> None:
    data = tomllib.loads((ROOT / "config.example.toml").read_text(encoding="utf-8"))
    assert "plugin" not in data
    assert data["plugin_runtime"]["enabled"] is True


def test_models_lock_covers_onnx_and_tokenizer() -> None:
    import json

    lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
    locals_ = {item["local"] for item in lock["files"]}
    assert {"model.onnx", "tokenizer.json"} <= locals_
    for item in lock["files"]:
        assert item["algo"] in {"sha256", "git-sha1"}
        assert len(item["digest"]) in {40, 64}


def test_i18n_files_cover_every_entry() -> None:
    """i18n 是扁平 key；漏一条入口文案，面板上就是空白。"""
    import json

    entries = _declared_entries()
    for name in ("zh-CN.json", "en.json"):
        data = json.loads((ROOT / "i18n" / name).read_text(encoding="utf-8"))
        assert "plugin.name" in data
        for entry_id in entries:
            assert f"entries.{entry_id}.name" in data, f"{name} 缺 entries.{entry_id}.name"


def test_static_panel_declared_and_present() -> None:
    """声明了面板却漏了文件，宿主只会给一个 404。"""
    ui = _manifest()["plugin"]["ui"]
    assert ui["enabled"] is True
    panels = ui["panel"]
    assert panels, "至少要有一个 panel"
    for panel in panels:
        target = ROOT / panel["entry"]
        assert target.is_file(), f"panel 入口不存在: {panel['entry']}"
        assert panel["entry"].endswith(".html") or panel["entry"].endswith(".md")


def test_panel_html_has_the_contract_hooks() -> None:
    """面板靠这些 id/接口工作，改名就等于把 UI 弄坏。"""
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for marker in ('id="state"', 'id="grid"', 'id="query"', 'id="upload"', "/runs", "trigger_response"):
        assert marker in html, f"面板缺少 {marker}"
