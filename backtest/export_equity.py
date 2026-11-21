#!/usr/bin/env python3
"""导出 v3 冻结口径的净值与回撤序列，供 rules.html 的「回撤曲线」浮窗直接绘制。

数字必须与 spec_v3.json 一致（同一引擎 engine_v2.simulate + V3_SPEC）。

    ../.venv-bt/bin/python export_equity.py
→ backtest/equity_data.js
"""
import json
from pathlib import Path

import numpy as np

import engine_v2 as E

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "equity_data.js"


def main():
    rows = E.load_features()
    for r in rows:
        r["score3"] = E.score_v3(r)
    dates, bench = E.load_calendar()
    closes, lot_price = E.prep(dates, rows)
    sc = {(r["code"], r["entry_date"]): r["score3"] for r in rows}

    tr = [r for r in rows if r["entry_date"] >= E.EVAL_START]
    sim = E.simulate(tr, dates, dict(E.V3_SPEC, closes=closes, lot_price=lot_price), sc)
    m = E.metrics(sim["eq"], dates, start=E.EVAL_START)
    st = E.trade_stats(sim["filled"])

    si = next(i for i, d in enumerate(dates) if d >= E.EVAL_START)
    eq = sim["eq"][si:] / sim["eq"][si]
    ds = dates[si:]
    b0 = np.array(bench[si:], float)
    b0 = b0 / b0[0]

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    trough_i = int(np.argmin(dd))
    peak_i = int(np.argmax(eq[:trough_i + 1])) if trough_i > 0 else 0

    longest, run = 0, 0
    for v in dd:
        run = run + 1 if v < 0 else 0
        longest = max(longest, run)

    data = {
        "meta": {
            "label": "v3 冻结口径 · 10 仓等额 · 仅强市开仓",
            "start": ds[0], "end": ds[-1],
            "total": float(m["total"]), "cagr": float(m["cagr"]),
            "mdd": float(m["mdd"]), "sharpe": float(m["sharpe"]), "vol": float(m["vol"]),
            "win": float(st["win"]), "avg": float(st["avg"]), "trades": int(st["n"]),
            "fills": int(sim["fills"]), "hold": float(np.mean([t["hold"] for t in tr])),
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
    OUT.write_text("window.EQUITY_DATA = " +
                   json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"→ {OUT}  {OUT.stat().st_size / 1024:.0f} KB  {len(ds)} 个交易日")
    print(f"  累计 {m['total']*100:+.2f}%  年化 {m['cagr']*100:+.2f}%  "
          f"最大回撤 {m['mdd']*100:.2f}%  Sharpe {m['sharpe']:.2f}  "
          f"成交 {sim['fills']} 笔  最长水下 {longest} 天")


if __name__ == "__main__":
    main()
