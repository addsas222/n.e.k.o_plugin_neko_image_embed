"""命令行自举 CLIP 权重（离线安装 / 排障用）。

    python tools/fetch_model.py --model-dir <状态目录>/data/models/clip-vit-base-patch32
    python tools/fetch_model.py --check          # 只校验，不下载

不传 ``--model-dir`` 时按宿主默认状态目录推断：
``%LOCALAPPDATA%/N.E.K.O/plugins/neko_image_embed/data/models/clip-vit-base-patch32``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[1]
# 注意：这里只能把**插件根**放进 sys.path，然后导入 `engine.*`。
# 绝不能写 `import neko_image_embed...` —— 那会执行插件根的 __init__.py，
# 而它 import plugin.sdk，在宿主之外必炸。
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from engine.embed.fetch import ensure_model, load_lock  # noqa: E402


def default_model_dir() -> Path:
    root = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(root) / "N.E.K.O" / "plugins" / "neko_image_embed" / "data" / "models" / "clip-vit-base-patch32"


def main() -> int:
    parser = argparse.ArgumentParser(description="下载并校验 CLIP ONNX 权重")
    parser.add_argument("--model-dir", default=str(default_model_dir()))
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--check", action="store_true", help="只校验，不下载")
    parser.add_argument("--lock", default=str(_PLUGIN_DIR / "models.lock.json"))
    args = parser.parse_args()

    lock = load_lock(Path(args.lock))
    if not lock:
        print(f"[错误] 清单不可读: {args.lock}")
        return 2
    print(f"模型: {lock.get('model')}  仓库: {lock.get('repo')}")

    state = ensure_model(
        Path(args.model_dir),
        lock_path=Path(args.lock),
        endpoint=args.endpoint,
        allow_download=not args.check,
        progress=lambda name, total: print(f"  {name}: {total / 1048576:.1f} MiB", end="\r"),
    )
    print()
    if state.ok:
        print(f"[完成] {state.detail}{'（本次有下载）' if state.downloaded else ''}")
        return 0
    print(f"[失败] {state.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
