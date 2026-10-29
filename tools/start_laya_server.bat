@echo off
rem 启动 laya 判断服务（SystemOne 协议），供插件「模型」判断层调用。
rem 需要装好 laya/torch 的解释器；默认用源码 venv。
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
"C:/Users/54301/N.E.K.O/.venv/Scripts/python.exe" -u "%~dp0laya_server.py" --port 8011
