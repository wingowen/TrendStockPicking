#!/usr/bin/env bash
# WSL 服务器版每日任务：国内网络直连东财，口径与本地完全一致。
# 构建完成后把数据 force-push 到 data 分支（单提交、不累积历史），
# 触发 GitHub Actions 的 deploy-panel 部署到 Pages。
set -euo pipefail
cd "$(dirname "$0")"

export TZ=Asia/Shanghai
export PATH="$HOME/.local/bin:$PATH"
PY="$PWD/.venv/bin/python"
mkdir -p logs
LOG="$PWD/logs/daily-$(date +%Y%m%d).log"

exec > >(tee -a "$LOG") 2>&1

echo "=== $(date '+%F %T') 开始"

# 东财会按 IP 限流：抓取失败率过高时退避重试，最多 3 轮
attempt=0
until (cd backtest && "$PY" -u fetch_data.py --force); do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 3 ]; then
    echo "=== 抓取连续失败 3 轮，放弃 $(date '+%F %T')"
    exit 1
  fi
  echo "=== 抓取失败，15 分钟后重试（$attempt/3）"
  sleep 900
done
(cd backtest && "$PY" -u fetch_extra.py)

# 板块主力净流入按天变化：清掉真实缓存强制重抓（路径与 board_flow.CACHE 一致），
# board_flow.load_real_flow 内部另有 1 天 TTL 兜底，防止漏清理时继续用旧数据
rm -f "$PWD/backtest/fflow_cache.json"
"$PY" -u build_panel_from_db.py --days 30

flow_cov=$(grep "真实覆盖" "$LOG" | tail -1 || true)
echo "${flow_cov:-[flow] 无覆盖率输出}"
case "$flow_cov" in
  *"真实覆盖 0/"*) echo "=== 警告：板块资金流全缺，今日板块打分退回代理口径（名单可能与本地不同）";;
esac
(cd backtest && "$PY" -u export_equity.py)
echo "=== 构建完成 $(date '+%F %T')"

# 数据用临时 index 打成一个无父提交，force-push 覆盖 data 分支：
# 远端只保留最新一版，仓库不会因每天 7MB 快照而膨胀。
export GIT_INDEX_FILE="$(mktemp)"
git read-tree --empty
git add -f breakout-data.js breakout-data backtest/equity_data.js
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -m "data $(date '+%F %T')")
git push -f origin "$commit:refs/heads/data"
rm -f "$GIT_INDEX_FILE"
unset GIT_INDEX_FILE

echo "=== 已推送 data 分支 $(date '+%F %T')"
