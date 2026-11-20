#!/bin/bash
# 双击启动面板刷新服务（只需启动一次，之后常驻后台）
# 服务只在每天 15:05 收盘后允许刷新今日清单
cd "$(dirname "$0")"
PY="$PWD/.venv-bt/bin/python"

if pgrep -f "panel_server.py" >/dev/null 2>&1; then
  echo "刷新服务已在运行（端口 8765）"
  exit 0
fi

nohup "$PY" panel_server.py > panel_server.log 2>&1 &
sleep 1
if pgrep -f "panel_server.py" >/dev/null 2>&1; then
  echo "已启动面板刷新服务，日志见 panel_server.log"
  echo "现在回到面板，点击右上角「刷新今日清单」即可。"
else
  echo "启动失败，请查看 panel_server.log"
fi
