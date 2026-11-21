#!/usr/bin/env python3
"""把全部信号的「当日可知因子」缓存成 CSV，供 engine_v2.py 反复做组合实验。

一次 gather 约 4-5 分钟（要建板块等权指数并逐日取快照），之后所有组合实验都读缓存，
不再重复计算。缓存含：大盘状态、板块评分/连续流入、站稳天数、距突破位、量比，
以及用基准出口（trail=25% / max_hold=120）模拟出的成交明细和 20 日前瞻收益。

用法：
    .venv-bt/bin/python build_features.py
输出：
    features.csv
"""
import csv
import json
from pathlib import Path

import numpy as np

import rank_signals as RS
import run_backtest as R

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "features.csv"

COLS = ["code", "name", "breakout_date", "signal_date", "entry_date", "entry",
        "level", "vol_ratio", "gain", "days_above", "regime", "industry",
        "sec_score", "sec_consec", "sec_flow5", "sec_mom20",
        "exit_date", "exit", "ret", "hold", "reason", "mfe", "mae",
        "fwd20", "bench20"]


def main():
    rows, cal, bench_close = RS.gather()
    pos = {d: i for i, d in enumerate(cal)}
    print(f"[rows] {len(rows)}")

    out = []
    for r in rows:
        b = r["_bars"]
        e = r["_entry_i"]
        fwd20 = bench20 = None
        if e + 20 < len(b["c"]):
            fwd20 = float(b["c"][e + 20] / b["c"][e] - 1)
            di = pos.get(r["entry_date"])
            if di is not None and di + 20 < len(bench_close):
                bench20 = float(bench_close[di + 20] / bench_close[di] - 1)
        out.append({
            "code": r["code"], "name": r.get("name", ""),
            "breakout_date": r["breakout_date"], "signal_date": r["signal_date"],
            "entry_date": r["entry_date"], "entry": r["entry"],
            "level": r["level"], "vol_ratio": r["vol_ratio"], "gain": r["gain"],
            "days_above": r["days_above"], "regime": r["regime"] or "",
            "industry": r.get("industry", ""),
            "sec_score": r.get("sec_score"), "sec_consec": r.get("sec_consec"),
            "sec_flow5": r.get("sec_flow5"), "sec_mom20": r.get("sec_mom20"),
            "exit_date": r["exit_date"], "exit": r["exit"], "ret": r["ret"],
            "hold": r["hold"], "reason": r["reason"], "mfe": r["mfe"], "mae": r["mae"],
            "fwd20": fwd20, "bench20": bench20,
        })

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        for r in out:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in COLS})

    meta = {"n": len(out), "start": min(r["entry_date"] for r in out),
            "end": max(r["exit_date"] for r in out), "cols": COLS}
    (ROOT / "features_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT.name}  {len(out)} 行")


if __name__ == "__main__":
    main()
