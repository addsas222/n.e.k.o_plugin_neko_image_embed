# 图像记忆库 `neko_image_embed`

把图片变成可检索的记忆：**合规/质量闸门 → 人设判定链 → 压缩 → CLIP 向量入库**。
聊天时角色可以检索并用图片回应。

## 流水线

```
图片字节
  │
  ├─ 1. guard     解码/尺寸/像素上限/EXIF 剥离/清晰度/熵/dHash 去重     ← 不调模型
  ├─ 2. perceive  Sensenova 识图 → CLIP 零样本兜底（caption / tags）    ← 视觉层
  ├─ 3. judge     模型 / 宿主槽位 / 启发式 三选一，失败降级到启发式      ← 判断层
  ├─ 4. compress  1280px / q82 / ≤512 KiB（压不进就报错，不硬发）
  └─ 5. embed     CLIP ViT-B/32 → 向量 → SQLite（float32 BLOB + numpy 余弦）
```

「拒收」是正常结果（`accepted=false` + 原因）；「编码器不可用 / 写库失败 / 压不下去」才是
`Err`。两者在返回值里严格分开，避免把故障伪装成「不收录」。

## 安装

源码在 `<N.E.K.O>\plugins\neko_image_embed`，但宿主**只从执行根加载代码**：

```
%LOCALAPPDATA%\N.E.K.O\.neko-plugin-installations\plugins\neko_image_embed\
```

同步（宿主需关闭，否则文件被锁）：

```bat
python plugins\_sync_source_to_installed.py neko_image_embed
```

`config/ data/ cache/` 是宿主状态目录，不参与同步。

## 首次运行

1. 启动宿主，插件会出现在列表里，`auto_start=true`。
2. startup 只建目录，**模型在后台预热**：首次会从 `hf-mirror.com` 下载
   CLIP ViT-B/32 量化权重（约 154 MB）到
   `plugins\neko_image_embed\data\models\clip-vit-base-patch32\`。
3. 手动预下载（推荐，避免宿主里等）：

```bat
python plugins\neko_image_embed\tools\fetch_model.py
python plugins\neko_image_embed\tools\fetch_model.py --check   # 只校验
```

首次做判定时还会把 30 条主题提示词编码一次（约 3–5 秒），结果写进
`cache\zeroshot\topics-<id>.npy`（61 KB），之后每次重载都直接读缓存。

权重哈希钉在 `models.lock.json`：ONNX 走 sha256，小文件走 git blob sha1。

## 配置要点

安装后改 `<N.E.K.O>\plugins\neko_image_embed\config\plugin.toml`（首次由
`config.example.toml` 复制）。常改的几项：

| 键 | 作用 |
|---|---|
| `[image_embed.judge].mode` | 判断层：`model` / `host` / `heuristic`（面板可切换） |
| `[image_embed.judge].perception` | 识图链，默认 `["sensenova","zeroshot"]` |
| `[image_embed].min_value` | 价值分下限，默认 0.45 |
| `[image_embed].max_risk` | 任一风险维度上限，默认 0.60 |
| `[image_embed.embedder].backend` | `auto` / `clip_onnx` / `caption_text` / `external` |
| `[image_embed].persona_summary` | 留空则读宿主 `config\characters.json` 的当前猫娘 |
| `[image_embed].watch_frames` | 定时轮询 `bus.frames` 自动收录，默认 false |
| `[image_embed.query_aliases]` | 中文查询别名（`猫 = "cat"`），弥补 CLIP 文本塔偏英文 |

### 识图与判断（两层，别混）

| 层 | 干什么 | 谁来做 |
|---|---|---|
| 识图 perception | 图 → caption / tags | **Sensenova**（云端视觉）；不可用时退本地 CLIP 零样本 |
| 判断 judgment | 识图结果 + 人设 → `should_embed` / `value` / `risk` | **三选一**，见下 |

判断层只有三种候选（面板上有对应的切换窗口）：

| mode | 含义 | 现状 |
|---|---|---|
| `model` | 一个判断模型端点（SystemOne 或 OpenAI 兼容；laya/kev/jev 都只是它的实现） | 端点未就绪时为“不可用”，自动降级 |
| `host` | 宿主模型槽位（`ctx.models` + `[plugin.models.judge]`） | 未绑槽位 → 403 → 降级 |
| `heuristic` | 纯启发式：质量 + 识图标签命中人设的轻加权 | **当前生效**，永远可用 |

结果里分开记：`perception=sensenova`、`engine=rules`（判断层）。判断失败只降级、不伪造，
错误逐条进 `errors`。

### 本地 kev 判断服务（模型层的可用实现）

宿主是 PyInstaller 冻结包，插件进程 import 不到第三方包；判断模型只能作为**同机 HTTP 服务**：

```bat
tools\start_kev_server.bat        :: 127.0.0.1:8009，kev-0.8b，CPU 约 5s/次
```

仓库在 `%LOCALAPPDATA%\N.E.K.O\_local_models\kev`（首次 `uv sync --extra serve`，
首次运行会下 Qwen3.5-0.8B 底座）。它讲 SystemOne 协议，插件直接指过去：

```toml
[image_embed.judge]
mode = "model"                       # 判断层切到「模型」

[image_embed.judge.model]
enabled = true
protocol = "systemone"
base_url = "http://127.0.0.1:8009"
model = "kev-latest"
```

实测（2026-10-29）：黑白小狗 → 收录 0.67；合成截图 → 不收录 0.47，区分度正常。
价值取的是 `should_embed` 里「收录」那一项的概率，不是五档 score（score 会落在 0.43
被 `min_value` 误杀）。

laya 也试过（0.4.17 + torch CPU，服务化跑通），但零样本路由严重偏负——猫娘看橘猫都判
`丢弃` p=0.986，所以没采用；它的包和模型还在，`tools/start_laya_server.bat` 可复现。

### 配 Sensenova 识图

```toml
[image_embed.judge]
mode = "heuristic"                        # 判断层：model / host / heuristic
perception = ["sensenova", "zeroshot"]    # 识图层

[image_embed.judge.sensenova]
enabled = true
protocol = "openai_chat"
base_url = "https://token.sensenova.cn/v1"
api_key = "sk-..."                        # 只写运行时 config
model = "sensenova-6.8-flash-lite"
vision = true
```

识图失败（端点不通/额度用完）会自动退到 CLIP 零样本，收录不会中断；
这时 caption 会退化成「可能是：狗」这类标签拼接。

## 面板

插件自带一个静态面板，宿主里点开即可：

```
http://127.0.0.1:<插件端口>/plugin/neko_image_embed/ui
```

三块功能：**选文件直接收录**（不必再拼 b64）、**检索**（中文可用，走别名 + 标签词法）、
**网格浏览**（缩略图、caption、标签、来源引擎，可一键发送/遗忘）。底部还有**运行日志**：
收录/发送/遗忘/清理逐条记录，带级别（INFO/WARN/ERR）、引擎、耗时，可按级别过滤。

- 面板是 `static/index.html`，无构建步骤；数据靠新入口 `img_board` 提供
  （最近/搜索结果 + 缩略图 data URL）。
- 调用走宿主通用的 `/runs`（POST → 轮询 → `/export`），带 CSRF token 引导与 403 重试。
- 为什么图片要用 data URL：插件数据目录没有对外静态路由，`/media/<id>` 只服务
  临时上传缓存（128 MiB LRU，会过期）。
- `img_board` 单次最多 24 条、每条缩略图 ≤400 KB，避免一次刷新把 payload 顶爆。
- 日志存在 `images` 库的 `events` 表（环形保留 500 条），入口是 `img_log`。
  它记的是**操作级**事件，不是调试日志；调试细节仍在
  `<N.E.K.O>\logs\plugin\N.E.K.O_Plugin_neko_image_embed_*.log`。

## 人设隔离与归属

**每个角色一套独立存档，物理隔离、互不引用**：

```
data/images/
├── YUI/
│   ├── index.db        # 只有 YUI 的元数据 + 向量 + 事件日志
│   ├── media/          # 只有 YUI 的图片文件
│   └── thumbs/
├── 沐莹/
│   └── ...
└── index.db.migrated-*  # 旧版「单库 + lanlan 列」的归档，只读留证
```

- 不是「一张表加一列」再过滤——那样归档仍共用，任何一处漏过滤就串档。现在是不同的
  `index.db`、不同的媒体目录：同一张图在两个角色下各存一份、各有各的 `value`/`caption`。
- **入库必须能确定当前角色**：显式 `lanlan` → `ctx.current_lanlan` → `characters.json` 的
  「当前猫娘」。三者都拿不到就**拒绝入库**（`NO_PERSONA`），不会悄悄落进公共桶。
- 检索、浏览、日志、GC 都只操作当前角色的存档；面板顶部「存档」下拉切换看哪一档，
  一次只看一档，不做跨档合并查询。
- 角色侧的 LLM 工具拿不到角色直接拒绝（`IMAGE_MEMORY_NO_PERSONA`）；按 id 发送只在本档里找，
  找不到就是 `NOT_FOUND`，不存在“报别人的 id 就能发”。
- `img_reassign` 是**跨档迁移**：把记录连同文件搬到另一个角色（必须显式给 `lanlan`），
  源档删除、目标档新建，id 会变。旧版遗留的未归属记录就是用它收回的。

「需求度」由判定链第一级给出：`laya`/`gateway` 的 prompt 里带当前人设，
`value` 就是「对这个人设有多少价值」，`reason` 会写明理由。注意两点残余：
`zeroshot`/`rules` 只能按主题与质量打分、不懂人设；人设文案改了以后旧图的
`value` 不会自动重算（目前需要重新收录或后续补一个 `rejudge`）。

## 数据布局

```
plugins\neko_image_embed\
├── config\plugin.toml          # 生效配置
├── cache\
│   └── zeroshot\topics-<id>.npy  # 零样本主题矩阵（30×512，只算一次）
└── data\
    ├── images\<角色>\index.db   # 每个角色一套：元数据 + 向量 + 事件
    ├── images\<角色>\media\      # ≤512 KiB，已去 EXIF/ICC
    ├── images\<角色>\thumbs\     # 256px
    └── models\clip-vit-base-patch32\
```

## 已知限制（都是刻意的）

1. **网关未绑定时没有真正的语义价值判断**。`zeroshot` 能给主题标签和 caption，
   但它不判断「对角色有没有价值」——结果里固定带 `issues=["persona_not_evaluated"]`，
   只在人设文本里出现同名主题时做一次轻加权。要真正的判断，得给 `[plugin.models.judge]`
   绑一个带 `image_input` 的槽位。
2. **插件拿不到用户发到聊天的原图**。宿主只把「已送给模型的画面」放进 `bus.frames`
   （保留 4 帧、有损）。所以 `frames` 只是入口之一，`path`/`url`/`b64` 更可靠。
3. **聊天里的图片 URL 会过期**。`ctx.images.upload()` 背后是进程内 128 MiB LRU，
   宿主重启或淘汰后旧气泡的图会 404；文本 caption 仍在。
4. **动图取第 0 帧**。`allow_animated=false` 时直接拒收动图（不静默丢帧）；
   置 true 才收，并在结果里标 `flattened=true`。
5. **换编码器必须重建索引**。`meta` 表记 `embedder_id` + `dim`，不一致直接拒绝开库。
6. **没有 UI 面板**。v1 用入口 + LLM 工具；面板留到下一版。

## 升级注意

宿主只在**首次安装**时把 `config.example.toml` 复制成 `config/plugin.toml`，
之后新增的配置键**不会**自动出现在已有安装里。本插件新增 `zeroshot` 级时踩过这个坑：
代码支持但 `judge.order` 没写，它就永远不会启用。所以：

- 新增能力要有默认行为兜底；
- 必须在自检里能发现——本插件会在 `img_status.warnings` 与日志里提示
  「judge.order 里没有 zeroshot」；
- 升级后手动把 `zeroshot` 加进 `judge.order`（新装自动包含）。

## 排障

先跑 `img_status`：

- `state=initializing`：后台还在下载/加载模型，等一会儿再查。
- `state=degraded`：看 `init_error`。常见是下载失败（换 `hf_endpoint` 或手工放权重）、
  或换过编码器导致 `EmbedderMismatch`。
- `judge.active` 只有 `gateway` 且 `counts` 一直不涨：网关没绑槽位，且本地引擎都没开，
  此时实际是 `rules` 在工作（`verdict.engine=rules` 会明说）。

两个已经踩过、容易误判的点：

1. **日志里的 `Plugin event-loop stall: entry=img_ingest, elapsed=6.1s`**
   不是 403 慢，而是 `openai` 包在本机**冷导入 4.95 秒**，且它是在 `ctx.models.get_client()`
   里懒加载的——也就是发生在事件循环上。现在已在 timer 里用 `asyncio.to_thread` 预热
   （`engine/judges/gateway.py::ensure_sdk_imported`）。再看到这条告警，先查是不是又
   引入了新的重导入。
2. **入口失败时 `error.code` 永远是 `INTERNAL`**
   宿主把 `Err` 转成 `PluginExecutionError`（`plugin/core/entry_executor.py:169-173`），
   只保留 `message`，我们传的 `code` 会被丢掉。所以**只能靠 message 分辨原因**，
   `code=` 仍然照写（未来宿主若透传就能用），但别指望在回包里看到。

自检：

```bat
python -m pytest -q                 :: 需要 PIL/numpy/pytest
python tests\stub_run.py            :: 桩 SDK 端到端，退出码 0 = 全过
ruff check .
```
