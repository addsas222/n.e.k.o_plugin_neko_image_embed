"""让插件源码在宿主之外可测。

``neko_image_embed/__init__.py`` 会 ``import plugin.sdk.plugin``，而宿主之外没有这棵树。
这里在收集前把插件根插到 ``sys.path``，并把 ``neko_image_embed`` 预注册成一个
「只有 __path__ 的命名空间包」，从而可以 ``import neko_image_embed.engine.guard``
而不触发那个 ``__init__``。

（同款做法见本机 ``agent_bridge/tests/conftest.py``，是这套 SDK 下最干净的范式。）
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
# 测试产物放 cache/ 下：插件目录同时是宿主状态根，而同步脚本只排除
# config/ data/ cache/ 与各类缓存目录。放别处会被整目录复制过去，或者根本写不进去
# （实测宿主根目录与 %TEMP% 都不可写）。
TMP_ROOT = PLUGIN_ROOT / "cache" / "test-tmp"

if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

if "neko_image_embed" not in sys.modules:
    package = types.ModuleType("neko_image_embed")
    package.__path__ = [str(PLUGIN_ROOT)]  # type: ignore[attr-defined]
    sys.modules["neko_image_embed"] = package


@pytest.fixture
def tmp_path() -> Path:
    """只创建、不删除的临时目录。

    内置 ``tmp_path`` 会在会话结束时 rmtree；本机的 safe-delete 钩子会把它搬到
    ``%TEMP%\\codebuddy-safe-delete-bulk``，而那个目录在当前环境下不可写，
    结果整个 pytest 进程被判定失败。这里改成一个不会被清理的独立目录。
    """
    base = TMP_ROOT / f"t{uuid.uuid4().hex[:12]}"
    base.mkdir(parents=True, exist_ok=False)
    return base
