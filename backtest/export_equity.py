#!/usr/bin/env python3
"""导出当前口径（6 仓 + 低开过滤 + 25% 移动止盈 + 120 日）的净值与回撤序列，
供 rules.html 的浮窗直接绘制，不用跳页。"""
import json
from pathlib import Path

import numpy as np

import run_backtest as rb
from run_backtest import BASE, EVAL_START, WARMUP_BARS, load_data, find_signals

ROOT = Path(__file__).resolve().parent


def main():
    cfg = dict(BASE)
    stocks, data, idx = load_data()
    bench = idx["沪深300"]
    dates = bench["dates"]

    trades = []
    for code, b in data.items():
        if len(b["dates"]) < WARMUP_BARS:
            continue
        for sig in find_signals(b, cfg):
            if b["dates"][sig["signal_idx"]] < EVAL_START:
                continue
            t = rb.simulate_trade(b, sig, cfg)
            if not t:
                continue
            t["code"] = code
            t["_bars"] = b
            t["_entry_i"] = sig["signal_idx"] + 1
            trades.append(t)
    trades.sort(key=lambda t: (t["entry_date"], t["code"]))

    eq = rb.build_equity(trades, dates, bench["close"], cfg)
    si = next(i for i, d in enumerate(dates) if d >= EVAL_START)
    eq = eq[si:]
    ds = dates[si:]
    eq = eq / eq[0]

    b0 = bench["close"][si:].astype(float)
    b0 = b0 / b0[0]

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    m = rb.metrics_from_equity(eq, ds)
    st = rb.trade_stats(trades)

    # 回撤区间：最大回撤的峰-谷
    trough_i = int(np.argmin(dd))
    peak_i = int(np.argmax(eq[:trough_i + 1])) if trough_i > 0 else 0
    # 最长回撤持续（水下最长连续天数）
    longest, run, best_span = 0, 0, (0, 0)
    for i, v in enumerate(dd):
        if v < 0:
            run += 1
            if run > longest:
                longest = run
                best_span = (i - run + 1, i)
        else:
            run = 0

    data = {
        "meta": {
            "label": "6 仓实盘口径 · 低开>3% 过滤",
            "start": ds[0], "end": ds[-1],
            "total": float(m["total"]), "cagr": float(m["cagr"]),
            "mdd": float(m["mdd"]), "sharpe": float(m["sharpe"]), "vol": float(m["vol"]),
            "win": float(st["win"]), "avg": float(st["avg"]), "trades": int(st["n"]),
            "fills": int(getattr(rb.build_equity, "last_fills", 0)),
            "hold": float(st["hold"]),
            "mdd_peak": ds[peak_i], "mdd_trough": ds[trough_i],
            "longest_underwater": longest,
            "current_dd": float(dd[-1]),
            "bench_total": float(b0[-1] - 1),
        },
        "dates": [d.replace("-", "") for d in ds],
        "eq": [round(float(v), 4) for v in eq],
        "bench": [round(float(v), 4) for v in b0],
        "dd": [round(float(v), 4) for v in dd],
    }
    out = ROOT / "equity_data.js"
    out.write_text("window.EQUITY_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"→ {out}  {out.stat().st_size/1024:.0f} KB  {len(ds)} 个交易日")
    print(f"  累计 {m['total']*100:+.2f}%  年化 {m['cagr']*100:+.2f}%  最大回撤 {m['mdd']*100:.2f}%  "
          f"Sharpe {m['sharpe']:.2f}  最长水下 {longest} 天")


if __name__ == "__main__":
    main()
