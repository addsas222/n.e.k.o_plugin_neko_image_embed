"""把 laya 包成一个本地 SystemOne 服务，给 neko_image_embed 的「模型」判断层用。

为什么要单独起服务：宿主是 PyInstaller 冻结包（projectneko_server.exe），插件进程
没有可 pip 的 Python，import 不到 laya。而 laya + torch 装在源码 venv 里，所以让它
在本机端口上说话，插件通过 HTTP 调——协议就是插件已有的 `/v1/systemone`。

启动（用装着 laya 的解释器）：
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
      C:/Users/54301/N.E.K.O/.venv/Scripts/python.exe tools/laya_server.py --port 8011

健康检查：GET /healthz
判断：POST /v1/systemone  {"state": "...", "questions": {...}}
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import laya

_LOCK = threading.Lock()
_ROUTER = None

# laya 是一条二分类路由：收录 / 不收录。阈值以下的置信度按「不收录」处理。
CRITERIA = {
    "收录": "与角色（人设、兴趣、日常）相关，值得作为记忆保留",
    "不收录": "与角色无关、无意义、重复或质量太差",
}
INSTRUCTIONS = "结合角色设定判断：这张图的描述值得为角色记住吗？"

_SCORE_LEVELS = ((0.85, "很高"), (0.7, "高"), (0.5, "中"), (0.3, "低"), (0.0, "很低"))


def router() -> laya.LayaRouter:
    global _ROUTER
    with _LOCK:
        if _ROUTER is None:
            _ROUTER = laya.LayaRouter(criteria=CRITERIA, instructions=INSTRUCTIONS)
        return _ROUTER


def _route_of(decision: object) -> tuple[str, float]:
    """laya 的 last_decision 是 dict：answers.route.choice + probabilities。"""
    if isinstance(decision, dict):
        route_ans = (decision.get("answers") or {}).get("route") or {}
        choice = str(route_ans.get("choice") or "")
        probs = route_ans.get("probabilities") or {}
        best = max((float(v) for v in probs.values() if isinstance(v, (int, float))), default=0.0)
        return choice, max(0.0, min(best, 1.0))
    value = getattr(decision, "confidence", None)
    try:
        return "", max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return "", 0.5


def _score_label(conf: float) -> str:
    for floor, label in _SCORE_LEVELS:
        if conf >= floor:
            return label
    return "很低"


def judge(state: str) -> dict[str, object]:
    engine = router()
    route = engine(state or "")
    choice, conf = _route_of(getattr(engine, "last_decision", None))
    if choice:
        route = choice
    remember = str(route).strip().startswith("收录")
    return {
        "answers": {
            "should_embed": "收录" if remember else "不收录",
            "value": _score_label(conf),
            "sticker": "否",
        },
        "route": route,
        "confidence": round(conf, 4),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:  # 安静点，别刷屏
        return

    def _send(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        if self.path.split("?", 1)[0] in {"/healthz", "/health"}:
            self._send({"ok": True, "engine": "laya", "loaded": _ROUTER is not None})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] not in {"/v1/systemone", "/systemone"}:
            self._send({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send({"error": "bad json"}, 400)
            return
        state = str(payload.get("state") or "")
        if not state:
            self._send({"error": "state 不能为空"}, 400)
            return
        try:
            result = judge(state)
        except Exception as exc:  # noqa: BLE001 - 服务端不吞栈，但要回结构化错误
            self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return
        self._send(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"laya server on http://{args.host}:{args.port} (model loads on first request)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
