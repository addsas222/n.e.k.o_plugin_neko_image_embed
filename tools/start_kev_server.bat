@echo off
rem kev 判断服务（SystemOne 协议，127.0.0.1:8009），供插件「模型」判断层调用。
rem 仓库在 %LOCALAPPDATA%\N.E.K.O\_local_models\kev，0.8B 版本 CPU 可跑（约 5s/次）。
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
cd /d "%LOCALAPPDATA%\N.E.K.O\_local_models\kev"
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009 --device cpu
