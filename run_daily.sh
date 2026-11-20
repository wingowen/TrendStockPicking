#!/bin/bash
# 每日收盘后一键更新：全量重抓日线（保证复权口径一致）→ 补行业/不复权价/指数 → 重生成面板
# 建议 15:35 之后运行：./run_daily.sh
set -e
cd "$(dirname "$0")"
PY="$PWD/.venv-bt/bin/python"

echo "[1/3] 抓取日线（后复权）"
(cd backtest && "$PY" fetch_data.py --force)

echo "[2/3] 补行业、不复权收盘、深证成指/创业板指"
(cd backtest && "$PY" fetch_extra.py)

echo "[3/3] 生成面板数据"
"$PY" build_panel_from_db.py --days 30

echo "完成：打开 breakout-panel.html 查看"
