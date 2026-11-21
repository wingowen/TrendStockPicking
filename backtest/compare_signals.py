#!/usr/bin/env python3
"""对拍：面板 build_panel_from_db.evaluate 与回测 run_backtest.find_signals 的信号集合。

同一份 market.db、同一套参数，比较「首次突破日」集合是否一致。
只做只读检查，不写任何产物。

用法：
    ../.venv-bt/bin/python compare_signals.py --limit 200
    ../.venv-bt/bin/python compare_signals.py            # 全量

参数说明：
  --limit N   只检查前 N 只股票（0 = 全量），用于快速冒烟。
  LOOKBACK L  突破位窗口：前 L 个交易日最高收盘（不含当日）。
  BREAK_WIN W 观察窗口：突破后 W 个交易日内需完成站稳。
  vol_min/max 突破日成交量 / 前 5 日均量的允许区间。
  hold_min    最少站稳天数；gain_min 最少涨幅（小数，0.03 = 3%）。
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))

from build_panel_from_db import (  # noqa: E402
    BREAK_WIN, GAIN_MIN, HISTORY_START, HOLD_MIN, LOOKBACK, VOL_MAX, VOL_MIN,
    evaluate, load,
)
from run_backtest import BASE, find_signals  # noqa: E402


def iso(d):
    """面板 ymd() 输出 20250625，DB/回测是 2025-06-25，统一成后者。"""
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and d.isdigit() else d


def panel_signal_pairs(b):
    """面板口径：{突破日: 最早信号日}，evaluate 无 reasons 才算有效信号。

    面板 evaluate 对同一个突破会在多个 k 上重复返回，取最早 k 才能和
    回测 find_signals 的「一次突破只记一次」对齐。
    """
    out = {}
    c = b["close"]
    for k in range(LOOKBACK + BREAK_WIN + 5, len(c)):
        got = evaluate(b, k, {})
        if not got:
            continue
        rec, reasons = got
        if reasons:
            continue
        bd, sd = iso(rec["breakout_date"]), b["dates"][k]
        if bd not in out or sd < out[bd]:
            out[bd] = sd
    return out


def backtest_signal_pairs(b):
    """回测口径的 (突破日, 信号日) 集合。"""
    rbars = {
        "c": np.asarray(b["close"], dtype=float),
        "v": np.asarray(b["vol"], dtype=float),
        "dates": b["dates"],
    }
    return {(s["breakout_date"], s["signal_date"]) for s in find_signals(rbars, BASE)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只检查前 N 只股票，0=全量")
    ap.add_argument("--since", default="2025-01-01",
                    help="只比较该日之后的突破（默认 2025-01-01，避开暖机窗口边界）")
    args = ap.parse_args()

    con = sqlite3.connect(ROOT / "backtest" / "market.db")
    stocks, bars, _raw, _idx = load(con)
    con.close()

    codes = [c for c in bars if not c.startswith("IDX_")]
    if args.limit:
        codes = codes[:args.limit]

    only_panel, only_backtest, both = set(), set(), set()
    bp_panel, bp_backtest = set(), set()
    for code in codes:
        b = bars[code]
        p = {(code, x, y) for x, y in panel_signal_pairs(b).items() if x >= args.since}
        r = {(code, x, y) for x, y in backtest_signal_pairs(b) if x >= args.since}
        only_panel |= p - r
        only_backtest |= r - p
        both |= p & r
        bp_panel |= {(code, x) for _code, x, _y in p}
        bp_backtest |= {(code, x) for _code, x, _y in r}

    print(f"CFG L={LOOKBACK} W={BREAK_WIN} vol=[{VOL_MIN},{VOL_MAX}] "
          f"hold={HOLD_MIN} gain={GAIN_MIN} base_key={sorted(BASE)}")
    print(f"股票 {len(codes)} 只 | 一致 {len(both)} | 仅面板 {len(only_panel)} | 仅回测 {len(only_backtest)}")
    print(f"仅看首次突破日：面板 {len(bp_panel)} | 回测 {len(bp_backtest)} | "
          f"差异 {len(bp_panel ^ bp_backtest)}")
    print("注：信号日差异来自面板取『窗口内最早仍有效突破』、回测按时间顺序消费信号；"
          "P2 抽公共 signals 模块时统一。")
    for label, s in (("仅面板", only_panel), ("仅回测", only_backtest)):
        for row in sorted(s)[:15]:
            print(f"  {label}: {row}")


if __name__ == "__main__":
    main()
