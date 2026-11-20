#!/usr/bin/env bash
# GitHub Actions 版每日构建：不依赖本地 .venv / market.db，每次从零全量抓取。
# 本地仍用 run_daily.sh（走 .venv-bt），两者互不影响。
set -euo pipefail
cd "$(dirname "$0")"

# GitHub runner 默认 UTC；build_panel 内部用 datetime.now() 判断「15:05 前跳过当日」，
# 不设 TZ 会把北京 15:20 误判成凌晨 7:20 从而丢掉当天快照。
export TZ="${TZ:-Asia/Shanghai}"

PY="${PYTHON:-python3}"

echo "[1/4] 全量抓日线（后复权）"
(cd backtest && "$PY" -u fetch_data.py --force)

echo "[2/4] 补行业 / 不复权收盘 / 指数 / 板块资金流"
(cd backtest && "$PY" -u fetch_extra.py)

echo "[3/4] 生成面板数据（最近 30 个交易日）"
"$PY" -u build_panel_from_db.py --days 30

echo "[4/4] 导出规则页净值/回撤数据"
(cd backtest && "$PY" -u export_equity.py)

echo "完成：breakout-data.js $(du -h breakout-data.js | cut -f1)，快照 $(ls breakout-data | wc -l | tr -d ' ') 份"
